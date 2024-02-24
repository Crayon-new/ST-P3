from PIL import Image

import torch
import torch.utils.data
import PIL
import os
import json
from pyquaternion import Quaternion
import numpy as np
import torch
import torchvision
from stp3.config import get_cfg
from stp3.utils.geometry import (
    resize_and_crop_image,
    update_intrinsics,
    calculate_birds_eye_view_parameters,
    convert_egopose_to_matrix_numpy,
    pose_vec2mat,
    mat2pose_vec,
    invert_matrix_egopose_numpy,
    get_global_pose
)

class SuscapeDataset(torch.utils.data.Dataset):
    camera_calib_path = "/home2/huangzj/github_respo/ST-P3/data/suscape/suscape_scenes/scene-000000/calib/camera"
    cameras = ['front_left', 'front', 'front_right', 'rear_left', 'rear', 'rear_right']
    scenes_path = ["/home2/huangzj/github_respo/ST-P3/data/suscape/scenes_eval.txt",
                   "/home2/huangzj/github_respo/ST-P3/data/suscape/scenes_train.txt"]
    def __init__(self, root_dir, is_train, cfg):
        self.root_dir = root_dir if root_dir is not None else "/home2/huangzj/github_respo/ST-P3/data/suscape/suscape_scenes"
        self.scenes_id = SuscapeDataset.get_scenes(SuscapeDataset.scenes_path[0])
        self.sequence_length = 3 # or ?
        self.receptive_field = 3
        self.frames = []
        self.id_to_scene = {}
        self.cfg = cfg
        self.normalise_image = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.augmentation_parameters = self.get_resizing_and_cropping_parameters()
        self.camera_intrinsic, self.camera_extrinsic = self.get_camera_calib()
        self.indices = self.get_indices()

    @staticmethod
    def get_scenes(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
            return [line.strip() for line in lines]

    def get_camera_calib(self):
        extrinsics = []
        intrinsics = []
        for camera in SuscapeDataset.cameras:
            calib_path = os.path.join(SuscapeDataset.camera_calib_path, camera+'.json')
            with open(calib_path, 'r') as f:
                data = json.load(f)
            extrinsics_np = np.array(data['extrinsic']).reshape(4, 4)
            intrinsic_np = np.array(data['intrinsic']).reshape(3, 3)
            intrinsic = torch.from_numpy(intrinsic_np).float()
            sensor_to_lidar = torch.from_numpy(np.linalg.inv(extrinsics_np)).float()
            top_crop = self.augmentation_parameters['crop'][1]
            left_crop = self.augmentation_parameters['crop'][0]
            intrinsic = update_intrinsics(
                intrinsic, top_crop, left_crop,
                scale_width=self.augmentation_parameters['scale_width'],
                scale_height=self.augmentation_parameters['scale_height']
            )

            intrinsics.append(intrinsic.unsqueeze(0).unsqueeze(0))
            extrinsics.append(sensor_to_lidar.unsqueeze(0).unsqueeze(0))
        intrinsics, extrinsics = (
            torch.cat(intrinsics, dim=1),
            torch.cat(extrinsics, dim=1)
        )
        return intrinsics, extrinsics

    def get_resizing_and_cropping_parameters(self):
        original_height, original_width = 1536, 2048
        final_height, final_width = self.cfg.IMAGE.FINAL_DIM

        resize_scale = 0.25
        resize_dims = (int(original_width * resize_scale), int(original_height * resize_scale))
        resized_width, resized_height = resize_dims

        crop_h = 80
        crop_w = int(max(0, (resized_width - final_width) / 2))
        # Left, top, right, bottom crops.
        crop = (crop_w, crop_h, crop_w + final_width, crop_h + final_height)

        if resized_width != final_width:
            print('Zero padding left and right parts of the image.')
        if crop_h + final_height != resized_height:
            print('Zero padding bottom part of the image.')

        return {'scale_width': resize_scale,
                'scale_height': resize_scale,
                'resize_dims': resize_dims,
                'crop': crop,
                }

    def get_input_data(self, id):
        images = []
        depths = []

        for camera in self.cameras:
            scene_id = self.id_to_scene[id]
            image_filename = os.path.join(self.root_dir, scene_id, "camera", camera, self.frames[id]+'.jpg')
            img = Image.open(image_filename)
            # Resize and cropc
            img = resize_and_crop_image(
                img, resize_dims=self.augmentation_parameters['resize_dims'], crop=self.augmentation_parameters['crop']
            )
            # Normalise image
            img = self.normalise_image(img)

            # depth ?
            images.append(img.unsqueeze(0).unsqueeze(0))
        images = torch.cat(images, dim=1)
        if len(depths) > 0:
            depths = torch.cat(depths, dim=1)
        return images, depths

    def get_future_egomotion(self, idx_t0):
        future_egomotion = np.eye(4, dtype=np.float32)
        if idx_t0 < len(self.frames) - 1:
            idx_t1 = idx_t0+1
            if self.id_to_scene[idx_t0] == self.id_to_scene[idx_t1]:
                t0_pose_matrix = self.get_ego_pose(idx_t0)
                t1_pose_matrix = self.get_ego_pose(idx_t1)
                future_egomotion = invert_matrix_egopose_numpy(t1_pose_matrix).dot(t0_pose_matrix)
                future_egomotion[3, :3] = 0.0
                future_egomotion[3, 3] = 1.0

        future_egomotion = torch.Tensor(future_egomotion).float()

        # Convert to 6DoF vector
        future_egomotion = mat2pose_vec(future_egomotion)
        return future_egomotion.unsqueeze(0)

    def get_ego_pose(self, idx):
        pose_file = os.path.join(self.root_dir, self.id_to_scene[idx], 'ego_pose', self.frames[idx]+'.json')
        with open(pose_file, 'r') as f:
            data = json.load(f)
        pose_matrix = np.eye(4)
        translation = np.array([float(data['x']), float(data['y']), float(data['z'])])
        rotation = Quaternion(axis=[1, 0, 0], angle=float(data['roll'])) * \
                   Quaternion(axis=[0, 1, 0], angle=float(data['pitch'])) * \
                   Quaternion(axis=[0, 0, 1], angle=float(data['azimuth']))
        pose_matrix[:3, :3] = rotation.rotation_matrix
        pose_matrix[:3, 3] = translation
        return pose_matrix


    def get_label(self):
        pass

    def get_indices(self):
        indices = []
        base_idx = 0
        for scenes_id in self.scenes_id:
            image_path = os.path.join(self.root_dir, scenes_id, "camera/front")
            files = os.listdir(image_path)
            files.sort()
            frames = [f[:-4] for f in files]
            scene_sz = len(files) - self.sequence_length + 1
            if scene_sz < 0:
                print("Scene {} has {} frames, which is less than the sequence length {}".format(scenes_id, len(files), self.sequence_length))

            for idx in range(scene_sz):
                indice = []
                for seq in range(self.sequence_length):
                    indice.append(base_idx + idx + seq)
                    self.id_to_scene[base_idx + idx + seq] = scenes_id
                indices.append(indice)
            self.frames.extend(frames)
            base_idx += len(files)
        return indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        """
        :param index:
        :return:
        """
        data = {}
        keys = ['image', 'intrinsics', 'extrinsics', 'depths',
                'segmentation', 'instance', 'centerness', 'offset', 'flow', 'pedestrian',
                'future_egomotion', 'hdmap', 'gt_trajectory', 'indices',
                ]
        for key in keys:
            data[key] = []

        instance_map = {}
        # Loop over all the frames in the sequence
        for index_t, cell in enumerate(self.indices[index]):
            if index_t>= self.receptive_field:
                in_pred = True
            else:
                in_pred = False

            if index_t < self.receptive_field:
                images, depth = self.get_input_data(cell)
                data['image'].append(images)
                data['depths'].append(depth)
                data['extrinsics'].append(self.camera_extrinsic)
                data['intrinsics'].append(self.camera_intrinsic)
            # segmentation, instance, pedestrian, instance_map = self.get_label()

            future_egomotion = self.get_future_egomotion(cell)
            data['future_egomotion'].append(future_egomotion)
            # fake filled here
            data['command'] = depth
            data['sample_trajectory'] = depth
            data['target_point'] = depth


        for key, value in data.items():
            if key in ['image', 'intrinsics', 'extrinsics', 'depths',
                       'future_egomotion']:
                if key == 'depths' and self.cfg.LIFT.GT_DEPTH is False:
                    continue
                data[key] = torch.cat(value, dim=0)
        return data

# Test
if __name__ == '__main__':
    cfg = get_cfg()
    suscape = SuscapeDataset("/home2/huangzj/github_respo/ST-P3/data/suscape/suscape_scenes", None, cfg)
    suscape.__getitem__(0)
    print(len(suscape))
