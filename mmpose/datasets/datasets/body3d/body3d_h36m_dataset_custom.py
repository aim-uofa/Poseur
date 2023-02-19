from .body3d_h36m_dataset import Body3DH36MDataset
import os.path as osp
import tempfile
import warnings
from collections import OrderedDict, defaultdict

import mmcv
import numpy as np
from mmcv import Config, deprecated_api_warning

from mmpose.core.evaluation import keypoint_mpjpe

from ...builder import DATASETS


@DATASETS.register_module()
class CustomBody3DH36MDataset(Body3DH36MDataset):
    """Human3.6M dataset for 3D human pose estimation.

    "Human3.6M: Large Scale Datasets and Predictive Methods for 3D Human
    Sensing in Natural Environments", TPAMI`2014.
    More details can be found in the `paper
    <http://vision.imar.ro/human3.6m/pami-h36m.pdf>`__.

    Human3.6M keypoint indexes::

        0: 'root (pelvis)',
        1: 'right_hip',
        2: 'right_knee',
        3: 'right_foot',
        4: 'left_hip',
        5: 'left_knee',
        6: 'left_foot',
        7: 'spine',
        8: 'thorax',
        9: 'neck_base',
        10: 'head',
        11: 'left_shoulder',
        12: 'left_elbow',
        13: 'left_wrist',
        14: 'right_shoulder',
        15: 'right_elbow',
        16: 'right_wrist'


    Args:
        ann_file (str): Path to the annotation file.
        img_prefix (str): Path to a directory where images are held.
            Default: None.
        data_cfg (dict): config
        pipeline (list[dict | callable]): A sequence of data transforms.
        dataset_info (DatasetInfo): A class containing all dataset info.
        test_mode (bool): Store True when building test or
            validation dataset. Default: False.
    """

    JOINT_NAMES = [
        'Root', 'RHip', 'RKnee', 'RFoot', 'LHip', 'LKnee', 'LFoot', 'Spine',
        'Thorax', 'NeckBase', 'Head', 'LShoulder', 'LElbow', 'LWrist',
        'RShoulder', 'RElbow', 'RWrist'
    ]

    # 2D joint source options:
    # "gt": from the annotation file
    # "detection": from a detection result file of 2D keypoint
    # "pipeline": will be generate by the pipeline
    SUPPORTED_JOINT_2D_SRC = {'gt', 'detection', 'pipeline'}

    # metric
    ALLOWED_METRICS = {'mpjpe', 'p-mpjpe', 'n-mpjpe'}

    def __init__(self,
                 ann_file,
                 img_prefix,
                 data_cfg,
                 pipeline,
                 dataset_info=None,
                 test_mode=False,
                 e2e=False):
        
        self.e2e = e2e
        self.scale_factor = 1.2
        if dataset_info is None:
            warnings.warn(
                'dataset_info is missing. '
                'Check https://github.com/open-mmlab/mmpose/pull/663 '
                'for details.', DeprecationWarning)
            cfg = Config.fromfile('configs/_base_/datasets/h36m.py')
            dataset_info = cfg._cfg_dict['dataset_info']

        super().__init__(
            ann_file,
            img_prefix,
            data_cfg,
            pipeline,
            dataset_info=dataset_info,
            test_mode=test_mode)
        
    def load_config(self, data_cfg):
        super().load_config(data_cfg)
        # h36m specific attributes
        self.joint_2d_src = data_cfg.get('joint_2d_src', 'gt')
        if self.joint_2d_src not in self.SUPPORTED_JOINT_2D_SRC:
            raise ValueError(
                f'Unsupported joint_2d_src "{self.joint_2d_src}". '
                f'Supported options are {self.SUPPORTED_JOINT_2D_SRC}')

        self.joint_2d_det_file = data_cfg.get('joint_2d_det_file', None)

        self.need_camera_param = data_cfg.get('need_camera_param', False)
        if self.need_camera_param:
            assert 'camera_param_file' in data_cfg
            self.camera_param = self._load_camera_param(
                data_cfg['camera_param_file'])

        # h36m specific annotation info
        ann_info = {}
        ann_info['use_different_joint_weights'] = False
        # action filter
        actions = data_cfg.get('actions', '_all_')
        self.actions = set(
            actions if isinstance(actions, (list, tuple)) else [actions])

        # subject filter
        subjects = data_cfg.get('subjects', '_all_')
        self.subjects = set(
            subjects if isinstance(subjects, (list, tuple)) else [subjects])

        self.ann_info.update(ann_info)

    def load_annotations(self):
        """Load data annotation."""
        data = np.load(self.ann_file)
        # get image info
        _imgnames = data['image_path']
        num_imgs = len(_imgnames)
        num_joints = self.ann_info['num_joints']
        # import pdb
        # pdb.set_trace()
        # if 'scale' in data:
        #     _scales = data['scale'].astype(np.float32)
        # else:
        #     _scales = np.zeros(num_imgs, dtype=np.float32)

        # if 'center' in data:
        #     _centers = data['center'].astype(np.float32)
        # else:
        #     _centers = np.zeros((num_imgs, 2), dtype=np.float32)

        if 'bbox_xywh' in data:
            bbox_xywh = data['bbox_xywh']
            _centers = np.stack([bbox_xywh[:, 0] + (bbox_xywh[:, 2]) / 2,
                                bbox_xywh[:, 1] + (bbox_xywh[:, 3]) / 2],
                            axis=1)
            
            _scales = self.scale_factor * np.max(
                bbox_xywh[:, 2:4], axis=1) / 200

        elif 'bbox_xyxy' in data:
            bbox_xyxy = data['bbox_xyxy']
            _centers = np.stack([(bbox_xyxy[:, 0] + bbox_xyxy[:, 2]) / 2,
                                (bbox_xyxy[:, 1] + bbox_xyxy[:, 3]) / 2],
                            axis=1)
            
            _scales = self.scale_factor * np.max(
                bbox_xyxy[:, 2:] - bbox_xyxy[:, :2], axis=1) / 200
        else:
            # raise NotImplementedError
            _centers = np.zeros((num_imgs, 2), dtype=np.float32)
            _scales = np.zeros(num_imgs, dtype=np.float32)

        # get 3D pose
        if 'keypoints3d' in data.keys():
            _joints_3d = data['keypoints3d'].astype(np.float32)
        else:
            _joints_3d = np.zeros((num_imgs, num_joints, 4), dtype=np.float32)

        # get 2D pose
        if 'keypoints2d' in data.keys():
            _joints_2d = data['keypoints2d'].astype(np.float32)
        else:
            _joints_2d = np.zeros((num_imgs, num_joints, 3), dtype=np.float32)

        data_info = {
            'imgnames': _imgnames,
            'joints_3d': _joints_3d,
            'joints_2d': _joints_2d,
            'scales': _scales,
            'centers': _centers,
        }

        # get 2D joints
        if self.joint_2d_src == 'gt':
            data_info['joints_2d'] = data_info['joints_2d']
        elif self.joint_2d_src == 'detection':
            data_info['joints_2d'] = self._load_joint_2d_detection(
                self.joint_2d_det_file)
            assert data_info['joints_2d'].shape[0] == data_info[
                'joints_3d'].shape[0]
            assert data_info['joints_2d'].shape[2] == 3
        elif self.joint_2d_src == 'pipeline':
            # joint_2d will be generated in the pipeline
            pass
        else:
            raise NotImplementedError(
                f'Unhandled joint_2d_src option {self.joint_2d_src}')

        return data_info

    @staticmethod
    def _parse_h36m_imgname(imgname):
        """Parse imgname to get information of subject, action and camera.

        A typical h36m image filename is like:
        S1_Directions_1.54138969_000001.jpg
        """
        subj, rest = osp.basename(imgname).split('_', 1)
        action, rest = rest.split('.', 1)
        camera, rest = rest.split('_', 1)

        return subj, action, camera

    def build_sample_indices(self):
        """Split original videos into sequences and build frame indices.

        This method overrides the default one in the base class.
        """

        # Group frames into videos. Assume that self.data_info is
        # chronological.
        video_frames = defaultdict(list)
        for idx, imgname in enumerate(self.data_info['imgnames']):
            subj, action, camera = self._parse_h36m_imgname(imgname)

            if '_all_' not in self.actions and action not in self.actions:
                continue

            if '_all_' not in self.subjects and subj not in self.subjects:
                continue

            video_frames[(subj, action, camera)].append(idx)

        # build sample indices
        sample_indices = []
        _len = (self.seq_len - 1) * self.seq_frame_interval + 1
        _step = self.seq_frame_interval
        for _, _indices in sorted(video_frames.items()):
            n_frame = len(_indices)

            if self.temporal_padding:
                # Pad the sequence so that every frame in the sequence will be
                # predicted.
                if self.causal:
                    frames_left = self.seq_len - 1
                    frames_right = 0
                else:
                    frames_left = (self.seq_len - 1) // 2
                    frames_right = frames_left
                for i in range(n_frame):
                    pad_left = max(0, frames_left - i // _step)
                    pad_right = max(0,
                                    frames_right - (n_frame - 1 - i) // _step)
                    start = max(i % _step, i - frames_left * _step)
                    end = min(n_frame - (n_frame - 1 - i) % _step,
                              i + frames_right * _step + 1)
                    sample_indices.append([_indices[0]] * pad_left +
                                          _indices[start:end:_step] +
                                          [_indices[-1]] * pad_right)
            else:
                seqs_from_video = [
                    _indices[i:(i + _len):_step]
                    for i in range(0, n_frame - _len + 1)
                ]
                sample_indices.extend(seqs_from_video)

        # reduce dataset size if self.subset < 1
        assert 0 < self.subset <= 1
        subset_size = int(len(sample_indices) * self.subset)
        start = np.random.randint(0, len(sample_indices) - subset_size + 1)
        end = start + subset_size

        return sample_indices[start:end]

    def _load_joint_2d_detection(self, det_file):
        """"Load 2D joint detection results from file."""
        joints_2d = np.load(det_file).astype(np.float32)

        return joints_2d

    def _load_camera_param(self, camera_param_file):
        """Load camera parameters from file."""
        return mmcv.load(camera_param_file)

    def get_camera_param(self, imgname):
        """Get camera parameters of a frame by its image name."""
        assert hasattr(self, 'camera_param')
        subj, _, camera = self._parse_h36m_imgname(imgname)
        return self.camera_param[(subj, camera)]


    def prepare_data(self, idx):
        """Get data sample."""
        data = self.data_info

        frame_ids = self.sample_indices[idx]
        assert len(frame_ids) == self.seq_len

        # get the 3D/2D pose sequence
        _joints_3d = data['joints_3d'][frame_ids]
        _joints_2d = data['joints_2d'][frame_ids]

        # get the image info
        _imgnames = data['imgnames'][frame_ids]
        _centers = data['centers'][frame_ids]
        _scales = data['scales'][frame_ids]
        if _scales.ndim == 1:
            _scales = np.stack([_scales, _scales], axis=1)

        target_idx = -1 if self.causal else int(self.seq_len) // 2

        # import pdb
        # pdb.set_trace()
        if self.e2e:
            results = {
                'image_file': osp.join(self.img_prefix, _imgnames[0].replace('/images', '')),
                'scale': _scales,
                'center': _centers,
                # 'joints_3d': _joints_3d[:, :, :3],
                # 'joints_3d_visible': _joints_3d[:, :, -1:],
                # 'joints_3d': _joints_2d[:, :, :2],
                'joints_3d': _joints_2d[:, :, :].reshape(17, 3),
                'joints_3d_visible': _joints_2d[:, :, -1:].reshape(17, 1),
            }
        else:
            results = {
                'input_2d': _joints_2d[:, :, :2],
                'input_2d_visible': _joints_2d[:, :, -1:],
                'input_3d': _joints_3d[:, :, :3],
                'input_3d_visible': _joints_3d[:, :, -1:],
                'target': _joints_3d[target_idx, :, :3],
                'target_visible': _joints_3d[target_idx, :, -1:],
                'image_paths': _imgnames,
                'target_image_path': _imgnames[target_idx],
                'scales': _scales,
                'centers': _centers,
            }

        if self.need_2d_label:
            results['target_2d'] = _joints_2d[target_idx, :, :2]

        if self.need_camera_param:
            _cam_param = self.get_camera_param(_imgnames[0])
            results['camera_param'] = _cam_param
            # get image size from camera parameters
            if 'w' in _cam_param and 'h' in _cam_param:
                results['image_width'] = _cam_param['w']
                results['image_height'] = _cam_param['h']

        return results
