log_level = 'INFO'
load_from = None
resume_from = None
dist_params = dict(backend='nccl')
workflow = [('train', 1)]
checkpoint_config = dict(interval=10)
evaluation = dict(interval=25, metric='PCKh', key_indicator='PCKh', rle_score=True)

optimizer = dict(
    type='AdamW',
    lr=1e-3,
    weight_decay=1e-4,
    paramwise_cfg = dict(
        custom_keys={
            # 'backbone': dict(lr_mult=0.1),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1),
            # 'query_embed': dict(lr_mult=0.5, decay_mult=1.0),
        },
    )
)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[255, 310])
total_epochs = 325

log_config = dict(
    interval=50, hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])

channel_cfg = dict(
    num_output_channels=16,
    dataset_joints=16,
    dataset_channel=list(range(16)),
    inference_channel=list(range(16)))

emb_dim = 256

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='Poseur',
    pretrained='torchvision://resnet50',
    backbone=dict(type='ResNet', norm_cfg = norm_cfg, depth=50, num_stages=4, out_indices=(0, 1, 2, 3)),
    neck=dict(
        type='ChannelMapper',
        # in_channels=[128, 256, 512],
        in_channels=[256, 512, 1024, 2048],
        kernel_size=1,
        out_channels=emb_dim,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
    ),
    keypoint_head=dict(
        type='PoseurHead',
        in_channels=512,
        num_queries=channel_cfg['dataset_joints'],
        num_reg_fcs=2,
        num_joints=channel_cfg['num_output_channels'],
        with_box_refine=True,
        loss_coord_enc=dict(type='RLELoss_poseur', use_target_weight=True),
        loss_coord_dec=dict(type='RLELoss_poseur', use_target_weight=True),
        # loss_coord_dec=dict(type='L1Loss', use_target_weight=True, loss_weight=5),
        loss_hp_keypoint=dict(type='JointsMSELoss', use_target_weight=True, loss_weight=10),
        # loss_coord_keypoint=dict(type='L1Loss', use_target_weight=True, loss_weight=1),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=emb_dim//2,
            normalize=True,
            offset=-0.5),
        transformer=dict(
            type='PoseurTransformer',
            query_pose_emb = True,
            embed_dims = emb_dim,
            num_joints=channel_cfg['num_output_channels'],
            encoder=dict(
                type='DetrTransformerEncoder_zero_layer',
                num_layers=0,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    ffn_cfgs = dict(
                        embed_dims=emb_dim,
                        ),
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention',
                        num_levels=4,
                        num_points=4,
                        embed_dims=emb_dim),
                    
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='DeformableDetrTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer_grouped',
                    num_joints=channel_cfg['num_output_channels'],
                    ffn_cfgs = dict(
                        embed_dims=emb_dim,
                        ),
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=emb_dim,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='MultiScaleDeformableAttention_post_value',
                            num_levels=4,
                            num_points=4,
                            embed_dims=emb_dim)
                    ],
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        as_two_stage=True,
        use_heatmap_loss=False,
    ),
    train_cfg=dict(image_size=[192, 256]),
    test_cfg = dict(
        image_size=[192, 256],
        flip_test=True,
        post_process='default',
        shift_heatmap=True,
        modulate_kernel=11)
)

data_cfg = dict(
    image_size=[256, 256],
    heatmap_size=[64, 64],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    use_gt_bbox=True,
    bbox_file=None,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownGetBboxCenterScale', padding=1.25),
    dict(type='TopDownRandomFlip', flip_prob=0.5),
    dict(
        type='TopDownHalfBodyTransform',
        num_joints_half_body=8,
        prob_half_body=0.3),
    dict(
        type='TopDownGetRandomScaleRotation', rot_factor=40, scale_factor=0.5),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    # dict(
    #     type='TopDownGenerateTarget',
    #     kernel=[(11, 11), (9, 9), (7, 7), (5, 5)],
    #     encoding='Megvii'),
    dict(
        target_type='wo_mask',
        type='TopDownGenerateCoordAndHeatMapTarget',
        encoding='MSRA',
        sigma=2),
    dict(
        type='Collect',
        keys=['img', 'coord_target', 'coord_target_weight', 'hp_target', 'hp_target_weight'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'bbox_score', 'flip_pairs'
        ]),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownGetBboxCenterScale', padding=1.25),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='Collect',
        keys=[
            'img',
        ],
        meta_keys=[
            'image_file', 'center', 'scale', 'rotation', 'bbox_score',
            'flip_pairs'
        ]),
]

test_pipeline = val_pipeline

data_root = 'data/mpii'
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    train=dict(
        type='TopDownMpiiDataset',
        ann_file=f'{data_root}/annotations/mpii_train.json',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=train_pipeline),
    val=dict(
        type='TopDownMpiiDataset',
        ann_file=f'{data_root}/annotations/mpii_val.json',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=val_pipeline),
    test=dict(
        type='TopDownMpiiDataset',
        ann_file=f'{data_root}/annotations/mpii_val.json',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=test_pipeline),
)

fp16 = dict(loss_scale='dynamic')
