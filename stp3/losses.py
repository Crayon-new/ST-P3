import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialRegressionLoss(nn.Module):
    def __init__(self, norm, ignore_index=255, future_discount=1.0):
        super(SpatialRegressionLoss, self).__init__()
        self.norm = norm
        self.ignore_index = ignore_index
        self.future_discount = future_discount

        if norm == 1:
            self.loss_fn = F.l1_loss
        elif norm == 2:
            self.loss_fn = F.mse_loss
        else:
            raise ValueError(f'Expected norm 1 or 2, but got norm={norm}')

    def forward(self, prediction, target, n_present=3):
        assert len(prediction.shape) == 5, 'Must be a 5D tensor'
        # ignore_index is the same across all channels
        mask = target[:, :, :1] != self.ignore_index
        if mask.sum() == 0:
            return prediction.new_zeros(1)[0].float()

        loss = self.loss_fn(prediction, target, reduction='none')

        # Sum channel dimension
        loss = torch.sum(loss, dim=-3, keepdim=True)

        seq_len = loss.shape[1]
        assert seq_len >= n_present
        future_len = seq_len - n_present
        future_discounts = self.future_discount ** torch.arange(1, future_len+1, device=loss.device, dtype=loss.dtype)
        discounts = torch.cat([torch.ones(n_present, device=loss.device, dtype=loss.dtype), future_discounts], dim=0)
        discounts = discounts.view(1, seq_len, 1, 1, 1)
        loss = loss * discounts

        return loss[mask].mean()


class SegmentationLoss(nn.Module):
    def __init__(self, class_weights, ignore_index=255, use_top_k=False, top_k_ratio=1.0, future_discount=1.0):
        super().__init__()
        self.class_weights = class_weights
        self.ignore_index = ignore_index
        self.use_top_k = use_top_k
        self.top_k_ratio = top_k_ratio
        self.future_discount = future_discount

    def forward(self, prediction, target, n_present=3):
        if target.shape[-3] != 1:
            raise ValueError('segmentation label must be an index-label with channel dimension = 1.')
        b, s, c, h, w = prediction.shape

        prediction = prediction.view(b * s, c, h, w)
        target = target.view(b * s, h, w)
        loss = F.cross_entropy(
            prediction,
            target,
            ignore_index=self.ignore_index,
            reduction='none',
            weight=self.class_weights.to(target.device),
        )

        loss = loss.view(b, s, h, w)

        assert s >= n_present
        future_len = s - n_present
        future_discounts = self.future_discount ** torch.arange(1, future_len+1, device=loss.device, dtype=loss.dtype)
        discounts = torch.cat([torch.ones(n_present, device=loss.device, dtype=loss.dtype), future_discounts], dim=0)
        discounts = discounts.view(1, s, 1, 1)
        loss = loss * discounts

        loss = loss.view(b, s, -1)
        if self.use_top_k:
            # Penalises the top-k hardest pixels
            k = int(self.top_k_ratio * loss.shape[2])
            loss, _ = torch.sort(loss, dim=2, descending=True)
            loss = loss[:, :, :k]

        return torch.mean(loss)

class HDmapLoss(nn.Module):
    def __init__(self, class_weights, training_weights, use_top_k, top_k_ratio, ignore_index=255):
        super(HDmapLoss, self).__init__()
        self.class_weights = class_weights
        self.training_weights = training_weights
        self.ignore_index = ignore_index
        self.use_top_k = use_top_k
        self.top_k_ratio = top_k_ratio

    def forward(self, prediction, target):
        loss = 0
        for i in range(target.shape[-3]):
            cur_target = target[:, i]
            b, h, w = cur_target.shape
            cur_prediction = prediction[:, 2*i:2*(i+1)]
            cur_loss = F.cross_entropy(
                cur_prediction,
                cur_target,
                ignore_index=self.ignore_index,
                reduction='none',
                weight=self.class_weights[i].to(target.device),
            )

            cur_loss = cur_loss.view(b, -1)
            if self.use_top_k[i]:
                k = int(self.top_k_ratio[i] * cur_loss.shape[1])
                cur_loss, _ = torch.sort(cur_loss, dim=1, descending=True)
                cur_loss = cur_loss[:, :k]
            loss += torch.mean(cur_loss) * self.training_weights[i]
        return loss

class DepthLoss(nn.Module):
    def __init__(self, class_weights=None, ignore_index=255):
        super(DepthLoss, self).__init__()
        self.class_weights = class_weights
        self.ignore_index = ignore_index

    def forward(self, prediction, target):
        b, s, n, d, h, w = prediction.shape

        prediction = prediction.view(b*s*n, d, h, w)
        target = target.view(b*s*n, h, w)
        loss = F.cross_entropy(
            prediction,
            target,
            ignore_index=self.ignore_index,
            reduction='none',
            weight=self.class_weights
        )
        return torch.mean(loss)


class ProbabilisticLoss(nn.Module):
    def __init__(self, method):
        super(ProbabilisticLoss, self).__init__()
        self.method = method

    def kl_div(self, present_mu, present_log_sigma, future_mu, future_log_sigma):
        var_future = torch.exp(2 * future_log_sigma)
        var_present = torch.exp(2 * present_log_sigma)
        kl_div = (
                present_log_sigma - future_log_sigma - 0.5 + (var_future + (future_mu - present_mu) ** 2) / (
                2 * var_present)
        )

        kl_loss = torch.mean(torch.sum(kl_div, dim=-1))
        return kl_loss

    def forward(self, output):
        if self.method == 'GAUSSIAN':
            present_mu = output['present_mu']
            present_log_sigma = output['present_log_sigma']
            future_mu = output['future_mu']
            future_log_sigma = output['future_log_sigma']

            kl_loss = self.kl_div(present_mu, present_log_sigma, future_mu, future_log_sigma)
        elif self.method == 'MIXGAUSSIAN':
            present_mu = output['present_mu']
            present_log_sigma = output['present_log_sigma']
            future_mu = output['future_mu']
            future_log_sigma = output['future_log_sigma']

            kl_loss = 0
            for i in range(len(present_mu)):
                kl_loss += self.kl_div(present_mu[i], present_log_sigma[i], future_mu[i], future_log_sigma[i])
        elif self.method == 'BERNOULLI':
            present_log_prob = output['present_log_prob']
            future_log_prob = output['future_log_prob']

            kl_loss = F.kl_div(present_log_prob, future_log_prob, reduction='batchmean', log_target=True)
        else:
            raise NotImplementedError


        return kl_loss

class Seg_edl_log_loss(nn.Module):
    def __init__(self, class_weights, use_top_k, top_k_ratio, max_epochs, iters_per_epoch, coef=0.06, num_classes=2, use_kl=True):
        super(Seg_edl_log_loss, self).__init__()
        self.max_epochs = max_epochs
        self.num_classes = num_classes
        self.iters_per_epoch = iters_per_epoch
        self.coef = coef
        self.class_weights = class_weights
        self.use_top_k = use_top_k
        self.top_k_ratio = top_k_ratio
        self.eps = 1e-10
        self.use_kl = use_kl

    @staticmethod
    def kl_divergence(alpha, device=None):
        ones =torch.ones(alpha.size(), dtype=torch.float32, device=device)
        #ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
        sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
        first_term = (
            torch.lgamma(sum_alpha)
            - torch.lgamma(alpha).sum(dim=1, keepdim=True)
            + torch.lgamma(ones).sum(dim=1, keepdim=True)
            - torch.lgamma(ones.sum(dim=1, keepdim=True))
        )
        second_term = (
            (alpha - ones)
            .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
            .sum(dim=1, keepdim=True)
        )
        kl = first_term + second_term
        return kl

    # cite: Evidential Deep Learning for Open Set Action Recognition
    def euc(self, alpha, S, target, annealing_coef):
        pred_scores, pred_cls = torch.max(alpha / S, 1, keepdim=True)
        uncertainty = self.num_classes / S
        acc_match = torch.eq(pred_cls, torch.argmax(target, 1, keepdim=True)).float()
        acc_uncertain = - pred_scores * torch.log(1 - uncertainty + self.eps)
        inacc_certain = - (1 - pred_scores) * torch.log(uncertainty + self.eps)
        euc_loss = annealing_coef * acc_match * acc_uncertain + (1 - annealing_coef) * (1 - acc_match) * inacc_certain
        return euc_loss

    def edl_loss(self, alpha, target, cur_iter, cur_epoch, class_weights):
        device=alpha.device
        S = torch.sum(alpha, dim=1, keepdim=True)

        A = torch.sum(target * (torch.log(S) - torch.log(alpha)), dim=1, keepdim=True)
        # different class_weights for diff classes
        A = A * class_weights

        den = self.max_epochs * self.iters_per_epoch
        numer = ((cur_epoch) * self.iters_per_epoch) + cur_iter
        annealing_coef = torch.min(torch.tensor(1.0, dtype=torch.float32),torch.tensor(numer / den, dtype=torch.float32)).to(device)

        kl_alpha = (alpha - 1) * (1 - target) + 1
        if self.use_kl:
            ass_kl = (torch.tensor([[self.coef]]).to(device)) * annealing_coef * Seg_edl_log_loss.kl_divergence(kl_alpha, device=device)
        else:
            ass_kl = self.euc(alpha, S, target, annealing_coef)
        return A + ass_kl

    def forward(self, predict, target, cur_iter, cur_epoch):
        b, s, c, h, w = predict.shape
        class_weights = self.class_weights[target].to(predict.device)
        class_weights = class_weights.view(b * s, 1, h, w)
        target = F.one_hot(target.squeeze(dim=2), num_classes=c)
        target = target.permute(0, 1, 4, 2, 3)
        predict = predict.view(b * s, c, h, w)
        target = target.view(b * s, c, h, w)

        evidence = F.softplus(predict)
        alpha = evidence + 1

        loss = self.edl_loss(alpha, target, cur_iter, cur_epoch, class_weights=class_weights)

        loss = loss.view(b, s, -1)
        if self.use_top_k:
            # Penalises the top-k hardest pixels
            k = int(self.top_k_ratio * loss.shape[2])
            loss, _ = torch.sort(loss, dim=2, descending=True)
            loss = loss[:, :, :k]

        return torch.mean(loss)