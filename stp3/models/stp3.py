import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

from stp3.models.encoder import Encoder
from stp3.models.temporal_model import TemporalModelIdentity, TemporalModel
from stp3.models.distributions import DistributionModule
from stp3.models.future_prediction import FuturePrediction
from stp3.models.variance_net import VarianceNet
from stp3.models.decoder import Decoder
from stp3.models.planning_model import Planning
from stp3.utils.network import pack_sequence_dim, unpack_sequence_dim, set_bn_momentum
from stp3.utils.geometry import calculate_birds_eye_view_parameters, VoxelsSumming, pose_vec2mat


class STP3(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        bev_resolution, bev_start_position, bev_dimension = calculate_birds_eye_view_parameters(
            self.cfg.LIFT.X_BOUND, self.cfg.LIFT.Y_BOUND, self.cfg.LIFT.Z_BOUND
        )
        self.bev_resolution = nn.Parameter(bev_resolution, requires_grad=False)
        self.bev_start_position = nn.Parameter(bev_start_position, requires_grad=False)
        self.bev_dimension = nn.Parameter(bev_dimension, requires_grad=False)

        self.encoder_downsample = self.cfg.MODEL.ENCODER.DOWNSAMPLE
        self.encoder_out_channels = self.cfg.MODEL.ENCODER.OUT_CHANNELS

        self.frustum = self.create_frustum()
        self.depth_channels, _, _, _ = self.frustum.shape
        self.discount = self.cfg.LIFT.DISCOUNT

        if self.cfg.TIME_RECEPTIVE_FIELD == 1:
            assert self.cfg.MODEL.TEMPORAL_MODEL.NAME == 'identity'

        # temporal block
        self.receptive_field = self.cfg.TIME_RECEPTIVE_FIELD
        self.n_future = self.cfg.N_FUTURE_FRAMES
        self.latent_dim = self.cfg.MODEL.DISTRIBUTION.LATENT_DIM

        # Spatial extent in bird's-eye view, in meters
        self.spatial_extent = (self.cfg.LIFT.X_BOUND[1], self.cfg.LIFT.Y_BOUND[1])
        self.bev_size = (self.bev_dimension[0].item(), self.bev_dimension[1].item())

        # Encoder
        self.encoder = Encoder(cfg=self.cfg.MODEL.ENCODER, D=self.depth_channels)

        self.mean_conv = nn.Conv2d(self.encoder_out_channels, self.encoder_out_channels , 1)
        self.sigma_block = VarianceNet(self.encoder_out_channels, 2*self.encoder_out_channels,  self.encoder_out_channels)

        # Temporal model
        temporal_in_channels = self.encoder_out_channels
        if self.cfg.MODEL.TEMPORAL_MODEL.INPUT_EGOPOSE:
            temporal_in_channels += 6
        if self.cfg.MODEL.TEMPORAL_MODEL.NAME == 'identity':
            self.temporal_model = TemporalModelIdentity(temporal_in_channels, self.receptive_field)
        elif cfg.MODEL.TEMPORAL_MODEL.NAME == 'temporal_block':
            self.temporal_model = TemporalModel(
                temporal_in_channels,
                self.receptive_field,
                input_shape=self.bev_size,
                start_out_channels=self.cfg.MODEL.TEMPORAL_MODEL.START_OUT_CHANNELS,
                extra_in_channels=self.cfg.MODEL.TEMPORAL_MODEL.EXTRA_IN_CHANNELS,
                n_spatial_layers_between_temporal_layers=self.cfg.MODEL.TEMPORAL_MODEL.INBETWEEN_LAYERS,
                use_pyramid_pooling=self.cfg.MODEL.TEMPORAL_MODEL.PYRAMID_POOLING,
            )
        else:
            raise NotImplementedError(f'Temporal module {self.cfg.MODEL.TEMPORAL_MODEL.NAME}.')

        self.future_pred_in_channels = self.temporal_model.out_channels
        if self.n_future > 0:
            # probabilistic sampling
            if self.cfg.PROBABILISTIC.ENABLED:
                # Distribution networks
                self.present_distribution = DistributionModule(
                    self.future_pred_in_channels,
                    self.latent_dim,
                    method=self.cfg.PROBABILISTIC.METHOD
                )

            # Future prediction
            self.future_prediction = FuturePrediction(
                in_channels=self.future_pred_in_channels,
                latent_dim=self.latent_dim,
                n_future=self.n_future,
                mixture=self.cfg.MODEL.FUTURE_PRED.MIXTURE,
                n_gru_blocks=self.cfg.MODEL.FUTURE_PRED.N_GRU_BLOCKS,
                n_res_layers=self.cfg.MODEL.FUTURE_PRED.N_RES_LAYERS,
            )

        # Decoder
        self.decoder = Decoder(
            in_channels=self.future_pred_in_channels,
            n_classes=len(self.cfg.SEMANTIC_SEG.VEHICLE.WEIGHTS),
            n_present=self.receptive_field,
            n_hdmap=len(self.cfg.SEMANTIC_SEG.HDMAP.ELEMENTS),
            predict_gate = {
                'perceive_hdmap': self.cfg.SEMANTIC_SEG.HDMAP.ENABLED,
                'predict_pedestrian': self.cfg.SEMANTIC_SEG.PEDESTRIAN.ENABLED,
                'predict_instance': self.cfg.INSTANCE_SEG.ENABLED,
                'predict_future_flow': self.cfg.INSTANCE_FLOW.ENABLED,
                'planning': self.cfg.PLANNING.ENABLED,
            }
        )

        # Cost function
        # Carla 128, Nuscenes 256
        if self.cfg.PLANNING.ENABLED:
            self.planning = Planning(cfg, self.encoder_out_channels, 6, gru_state_size=self.cfg.PLANNING.GRU_STATE_SIZE)

        set_bn_momentum(self, self.cfg.MODEL.BN_MOMENTUM)

    def create_frustum(self):
        # Create grid in image plane
        h, w = self.cfg.IMAGE.FINAL_DIM
        downsampled_h, downsampled_w = h // self.encoder_downsample, w // self.encoder_downsample

        # Depth grid
        depth_grid = torch.arange(*self.cfg.LIFT.D_BOUND, dtype=torch.float)
        depth_grid = depth_grid.view(-1, 1, 1).expand(-1, downsampled_h, downsampled_w)
        n_depth_slices = depth_grid.shape[0]

        # x and y grids
        x_grid = torch.linspace(0, w - 1, downsampled_w, dtype=torch.float)
        x_grid = x_grid.view(1, 1, downsampled_w).expand(n_depth_slices, downsampled_h, downsampled_w)
        y_grid = torch.linspace(0, h - 1, downsampled_h, dtype=torch.float)
        y_grid = y_grid.view(1, downsampled_h, 1).expand(n_depth_slices, downsampled_h, downsampled_w)

        # Dimension (n_depth_slices, downsampled_h, downsampled_w, 3)
        # containing data points in the image: left-right, top-bottom, depth
        frustum = torch.stack((x_grid, y_grid, depth_grid), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def forward(self, image, intrinsics, extrinsics, future_egomotion):
        output = {}

        # Only process features from the past and present
        image = image[:, :self.receptive_field].contiguous() # (B2, F3, V6, C3, H224, W480) for nus
        intrinsics = intrinsics[:, :self.receptive_field].contiguous() # (B2, F3, V6, 3, 3) for nus
        extrinsics = extrinsics[:, :self.receptive_field].contiguous() # (B2, F3, V6, 4, 4) for nus
        future_egomotion = future_egomotion[:, :self.receptive_field].contiguous() #(2, 3, 6)

        # Lifting features and project to bird's-eye view
        x, depth, cam_front = self.calculate_birds_eye_view_features(image, intrinsics, extrinsics, future_egomotion) # (3,3,64,200,200)
        output = {**output, 'depth_prediction': depth, 'cam_front':cam_front}

        if self.cfg.MODEL.TEMPORAL_MODEL.INPUT_EGOPOSE:
            b, s, c = future_egomotion.shape
            h, w = x.shape[-2:]
            future_egomotions_spatial = future_egomotion.view(b, s, c, 1, 1).expand(b, s, c, h, w)
            # at time 0, no egomotion so feed zero vector
            future_egomotions_spatial = torch.cat([torch.zeros_like(future_egomotions_spatial[:, :1]),
                                                   future_egomotions_spatial[:, :(self.receptive_field-1)]], dim=1)
            x = torch.cat([x, future_egomotions_spatial], dim=-3)

        #  Temporal model
        states = self.temporal_model(x) # include 3 past feature

        if self.n_future > 0:
            present_state = states[:, -1:].contiguous()

            b, _, c, h, w = present_state.shape

            if self.cfg.PROBABILISTIC.ENABLED:
                sample = self.distribution_forward(
                    present_state,
                    min_log_sigma=self.cfg.MODEL.DISTRIBUTION.MIN_LOG_SIGMA,
                    max_log_sigma=self.cfg.MODEL.DISTRIBUTION.MAX_LOG_SIGMA,
                )
                future_prediction_input = sample
            else:
                future_prediction_input = present_state.new_zeros(b, 1, self.latent_dim, h, w)

            # predict the future
            states = self.future_prediction(future_prediction_input, states) #(2, 9, 64, 200, 200)

            # predict BEV outputs
            bev_output = self.decoder(states)
            output = {**output, **bev_output}

        else:
            b, n, c, h, w = states.shape
            states = states.view(b * n, c, h, w)
            mean_states = self.mean_conv(states)
            sigma_states = torch.clamp(self.sigma_block(states), -7, 7)
            # Perceive BEV outputs
            if self.cfg.MODEL.SAMPLE_RESULTS or True:
                # start = time.time()
                bev_output, heat_map= self.get_sample_results(mean_states, sigma_states, b, n)
                # end = time.time()
                # execution_time = end - start
                # print(f"执行时间: {execution_time} 秒")
                output = {**output, **bev_output, "mean_states": mean_states, "sigma_states": sigma_states, "heat_map": heat_map}
            else:
                bev_output = []
                for _ in range(self.cfg.MODEL.TRAIN_SAMPLE_NUM):
                    states = self.sample_with_uncertainty(mean_states, sigma_states, b, n)
                    bev_output.append(self.decoder(states))
                output = {**output, "aggregated_bev_output": bev_output, "mean_states": mean_states, "sigma_states": sigma_states}

        return output

    def get_geometry(self, intrinsics, extrinsics):
        """Calculate the (x, y, z) 3D position of the features.
        """
        rotation, translation = extrinsics[..., :3, :3], extrinsics[..., :3, 3]
        B, N, _ = translation.shape
        # Add batch, camera dimension, and a dummy dimension at the end
        points = self.frustum.unsqueeze(0).unsqueeze(0).unsqueeze(-1)

        # Camera to ego reference frame
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3], points[:, :, :, :, :, 2:3]), 5)
        combined_transformation = rotation.matmul(torch.inverse(intrinsics))
        points = combined_transformation.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += translation.view(B, N, 1, 1, 1, 3)

        # The 3 dimensions in the ego reference frame are: (forward, sides, height)
        return points

    def encoder_forward(self, x, cam_front_index=1):
        # batch, n_cameras, channels, height, width
        b, n, c, h, w = x.shape

        x = x.view(b * n, c, h, w) # (9 * 6, 3, 224, 480)
        x, depth = self.encoder(x) # (9 * 6, 64, 28, 60)
        if self.cfg.PLANNING.ENABLED:
            cam_front = x.view(b, n, *x.shape[1:])[:, cam_front_index]
        else:
            cam_front = None

        if self.cfg.MODEL.ENCODER.USE_DEPTH_DISTRIBUTION:
            depth_prob = depth.softmax(dim=1)
            x = depth_prob.unsqueeze(1) * x.unsqueeze(2)  # outer product depth and features
        else:
            x = x.unsqueeze(2).repeat(1, 1, self.depth_channels, 1, 1)

        x = x.view(b, n, *x.shape[1:])
        x = x.permute(0, 1, 3, 4, 5, 2) # channel dimension
        depth = depth.view(b, n, *depth.shape[1:])

        return x, depth, cam_front

    def projection_to_birds_eye_view(self, x, geometry, future_egomotion):
        """ Adapted from https://github.com/nv-tlabs/lift-splat-shoot/blob/master/src/models.py#L200"""
        # batch, s, n_cameras, depth, height, width, channels
        batch, s, n, d, h, w, c = x.shape
        output = torch.zeros(
            (batch, s, c, self.bev_dimension[0], self.bev_dimension[1]), dtype=torch.float, device=x.device
        )

        future_egomotion_mat = pose_vec2mat(future_egomotion)  # (3,3,4,4)
        rotation, translation = future_egomotion_mat[..., :3, :3], future_egomotion_mat[..., :3, 3]

        def voxel_to_pixel(geometry_b, x_b):
            # Mask out points that are outside the considered spatial extent.
            mask = (
                    (geometry_b[:, 0] >= 0)
                    & (geometry_b[:, 0] < self.bev_dimension[0])
                    & (geometry_b[:, 1] >= 0)
                    & (geometry_b[:, 1] < self.bev_dimension[1])
                    & (geometry_b[:, 2] >= 0)
                    & (geometry_b[:, 2] < self.bev_dimension[2])
            )
            x_b = x_b[mask]
            geometry_b = geometry_b[mask]

            # Sort tensors so that those within the same voxel are consecutives.
            ranks = (
                    geometry_b[:, 0] * (self.bev_dimension[1] * self.bev_dimension[2])
                    + geometry_b[:, 1] * (self.bev_dimension[2])
                    + geometry_b[:, 2]
            )
            ranks_indices = ranks.argsort()
            x_b, geometry_b, ranks = x_b[ranks_indices], geometry_b[ranks_indices], ranks[ranks_indices]

            # Project to bird's-eye view by summing voxels.
            x_b, geometry_b = VoxelsSumming.apply(x_b, geometry_b, ranks)
            return geometry_b, x_b

        # Number of 3D points
        N = n * d * h * w
        for b in range(batch):
            flow_b = x[b]
            flow_geo = geometry[b]

            #####  transform the 3D voxel to current frame  #####
            for t in range(s):
                if t != s - 1:
                    flow_geo_tmp = flow_geo[:t + 1]
                    rotation_b = rotation[b, t].view(1, 1, 1, 1, 1, 3, 3)
                    translation_b = translation[b, t].view(1, 1, 1, 1, 1, 3)
                    flow_geo_tmp = rotation_b.matmul(flow_geo_tmp.unsqueeze(-1)).squeeze(-1)
                    flow_geo_tmp += translation_b
                    flow_geo[:t + 1] = flow_geo_tmp

            bev_feature = torch.zeros((self.bev_dimension[2], self.bev_dimension[0], self.bev_dimension[1], c),
                                      device=flow_b.device)

            for t in range(s):
                # flatten x
                x_b = flow_b[t].reshape(N, c)

                # Convert positions to integer indices
                geometry_b = (
                        (flow_geo[t] - (self.bev_start_position - self.bev_resolution / 2.0)) / self.bev_resolution)
                geometry_b = geometry_b.view(N, 3).long()
                geometry_b, x_b = voxel_to_pixel(geometry_b, x_b)

                tmp_bev_feature = torch.zeros((self.bev_dimension[2], self.bev_dimension[0], self.bev_dimension[1], c),
                                              device=flow_b.device)
                tmp_bev_feature[geometry_b[:, 2], geometry_b[:, 0], geometry_b[:, 1]] = x_b

                bev_feature = bev_feature * self.discount + tmp_bev_feature
                tmp_bev_feature = bev_feature.permute((0, 3, 1, 2))
                tmp_bev_feature = tmp_bev_feature.squeeze(0)
                output[b, t] = tmp_bev_feature

        return output

    def calculate_birds_eye_view_features(self, x, intrinsics, extrinsics, future_egomotion):
        b, s, n, c, h, w = x.shape
        # Reshape
        x = pack_sequence_dim(x)
        intrinsics = pack_sequence_dim(intrinsics)
        extrinsics = pack_sequence_dim(extrinsics)

        geometry = self.get_geometry(intrinsics, extrinsics)
        x, depth, cam_front = self.encoder_forward(x)
        x = unpack_sequence_dim(x, b, s)
        geometry = unpack_sequence_dim(geometry, b, s)
        depth = unpack_sequence_dim(depth, b, s)
        cam_front = unpack_sequence_dim(cam_front, b, s)[:,-1] if cam_front is not None else None

        x = self.projection_to_birds_eye_view(x, geometry, future_egomotion)
        return x, depth, cam_front

    def sample_with_uncertainty(self, mean_states, sigma_states, b, n):
        if self.training:
            sample_states = mean_states + sigma_states.mul(0.5).exp_() * torch.randn_like(mean_states)
        else:
            sample_states = mean_states
        sample_states = sample_states.view(b, n, *sample_states.shape[1:])
        return sample_states

    def get_sample_results(self, mean_states, sigma_states, b, n):
        # just for uncertainty visualization
        sample = [np.zeros((200, 200)) for i in range(3)]
        # bev_output = self.decoder(states)
        sample_states = self.sample_with_uncertainty(mean_states, sigma_states, b, n)
        for i in range(self.cfg.MODEL.TEST_SAMPLE_NUM):
            sample_states = self.sample_with_uncertainty(mean_states, sigma_states, b, n)
            bev_output = self.decoder(sample_states)
            segmentation = bev_output['segmentation'][:, self.receptive_field-1].detach() # the present timestamp
            pedestrain = bev_output['pedestrian'][:, self.receptive_field-1].detach()
            hdmap = bev_output['hdmap'].detach()
            # drivable
            drivable_area = torch.argmax(hdmap[0, 2:4], dim=0).cpu().numpy()
            # lane
            lane_area = torch.argmax(hdmap[0, 0:2], dim=0).cpu().numpy()
            # car
            semantic_seg_car = torch.argmax(segmentation[0], dim=0).cpu().numpy()
            # pedestrain
            semantic_seg_pedes = torch.argmax(pedestrain[0], dim=0).cpu().numpy()

            drivable_area_index = drivable_area > 0
            lane_area_index = lane_area > 0
            semantic_seg_car_index = semantic_seg_car > 0
            semantic_seg_pedes_index = semantic_seg_pedes > 0
            sample[0][drivable_area_index] = sample[0][drivable_area_index]+1
            # sample[1][lane_area_index] = sample[1][lane_area_index]+1
            sample[1][semantic_seg_car_index] = sample[1][semantic_seg_car_index]+1
            sample[2][semantic_seg_pedes_index] = sample[2][semantic_seg_pedes_index]+1

        sample = [i/self.cfg.MODEL.TEST_SAMPLE_NUM for i in sample]
        index = [np.logical_or(i  == 0, i  == 1) for i in sample]
        entropy = np.zeros((200, 200))

        for idx, sp in zip(index, sample):
            entropy[~idx] = entropy[~idx] - (sp[~idx] * np.log2(sp[~idx]) + (1 - sp[~idx]) * np.log2(1 - sp[~idx]))

        return bev_output, entropy

    def distribution_forward(self, present_features, min_log_sigma, max_log_sigma):
        """
        Parameters
        ----------
            present_features: 5-D output from dynamics module with shape (b, 1, c, h, w)

        Returns
        -------
            sample: sample taken from present/future distribution, broadcast to shape (b, s, latent_dim, h, w)
        """
        b, s, _, h, w = present_features.size()
        assert s == 1

        def get_mu_sigma(mu_log_sigma):
            mu = mu_log_sigma[:, :, :self.latent_dim]
            log_sigma = mu_log_sigma[:, :, self.latent_dim:2*self.latent_dim]
            log_sigma = torch.clamp(log_sigma, min_log_sigma, max_log_sigma)
            sigma = torch.exp(log_sigma)
            if self.training:
                gaussian_noise = torch.randn((b, s, self.latent_dim), device=present_features.device)
            else:
                gaussian_noise = torch.zeros((b, s, self.latent_dim), device=present_features.device)
            sample = mu + sigma * gaussian_noise
            return mu, log_sigma, sample


        if self.cfg.PROBABILISTIC.METHOD == 'GAUSSIAN':
            mu_log_sigma = self.present_distribution(present_features)
            present_mu, present_log_sigma, present_sample = get_mu_sigma(mu_log_sigma)
            sample = present_sample

            # Spatially broadcast sample to the dimensions of present_features
            sample = sample.view(b, s, self.latent_dim, 1, 1).expand(b, s, self.latent_dim, h, w)

        elif self.cfg.PROBABILISTIC.METHOD == "BERNOULLI":
            present_log_prob = self.present_distribution(present_features)
            if self.training:
                bernoulli_noise = torch.randn((b, self.latent_dim, h, w), device=present_features.device)
            else:
                bernoulli_noise = torch.zeros((b, self.latent_dim, h, w), device=present_features.device)
            sample = torch.exp(present_log_prob) + bernoulli_noise

            sample = sample.view(b, s, self.latent_dim, h, w)


        elif self.cfg.PROBABILISTIC.METHOD == 'MIXGAUSSIAN':
            mu_log_sigma = self.present_distribution(present_features)
            present_mu1, present_log_sigma1, present_sample1 = get_mu_sigma(mu_log_sigma[:, :, :2*self.latent_dim])
            present_mu2, present_log_sigma2, present_sample2 = get_mu_sigma(mu_log_sigma[:, :, 2 * self.latent_dim : 4 * self.latent_dim])
            present_mu3, present_log_sigma3, present_sample3 = get_mu_sigma(mu_log_sigma[:, :, 4 * self.latent_dim : 6 * self.latent_dim])
            coefficient = mu_log_sigma[:, :, 6 * self.latent_dim:]
            coefficient = torch.softmax(coefficient, dim=-1)
            sample = present_sample1 * coefficient[:,:,0:1] + \
                     present_sample2 * coefficient[:,:,1:2] + \
                     present_sample3 * coefficient[:,:,2:3]

            # Spatially broadcast sample to the dimensions of present_features
            sample = sample.view(b, s, self.latent_dim, 1, 1).expand(b, s, self.latent_dim, h, w)

        else:
            raise NotImplementedError

        return sample

    def select_best_traj(self, trajs, cost_volume, lane_divider, semantic_pred, k=1):
        '''
        trajs: torch.Tensor (B, N, n_future, 3)
        fuser_feature: torch.Tensor (B, n_present, fuser_channel, 200, 200)
        hd_map: torch.Tensor(B, 5, 200, 200)
        semantic_pred: torch.Tensor(B, n_future, 200, 200)
        '''
        sm_cost_fc, sm_cost_fo = self.cost_function(cost_volume, trajs[:,:,:,:2], semantic_pred, lane_divider)

        CS = sm_cost_fc + sm_cost_fo.sum(dim=-1)
        CC, KK = torch.topk(CS, k, dim=-1, largest=False)

        ii = torch.arange(len(trajs))
        select_traj = trajs[ii[:,None], KK].squeeze(1) # (B, n_future, 3)

        return select_traj
