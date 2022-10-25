# PORT=29500 CUDA_VISIBLE_DEVICES=0 ./tools/dist_train.sh configs/body/2d_kpt_sview_rgb_img/poseur/coco/poseur_res50_coco_256x192.py 1 --work-dir work_dirs/poseur_res50_coco_256x192 --cfg-options data.train_dataloader.samples_per_gpu=8 data.workers_per_gpu=0
# 2022-10-24 17:07:43,436 - mmpose - INFO - Epoch [1][50/18727]   lr: 9.890e-05, eta: 35 days, 16:53:22, time: 0.507, data_time: 0.188, memory: 2977, enc_rle_loss: 74.6519, dec_rle_loss_0: 153.0398, dec_rle_loss_1: 162.1735, dec_rle_loss_2: 156.4055, dec_rle_loss_3: 149.5992, dec_rle_loss_4: 160.0592, dec_rle_loss_5: 163.5340, enc_coord_acc: 0.0004, dec_coord_acc: 0.0136, loss: 1019.4632

# PORT=29500 CUDA_VISIBLE_DEVICES=0 ./tools/dist_train.sh configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrformer_base_coco_256x192.py 1 --work-dir work_dirs/hrformer_base_coco_256x192 --cfg-options data.samples_per_gpu=8 data.workers_per_gpu=0
# 2022-10-24 17:05:16,173 - mmpose - INFO - Epoch [1][50/18727]   lr: 4.945e-05, eta: 34 days, 8:15:11, time: 0.755, data_time: 0.192, memory: 6237, heatmap_loss: 0.0048, acc_pose: 0.0139, loss: 0.0048

# PORT=29500 CUDA_VISIBLE_DEVICES=0 ./tools/dist_train.sh configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hourglass52_coco_256x256.py 1 --work-dir work_dirs/hourglass52_coco_256x256 --cfg-options data.train_dataloader.samples_per_gpu=8 data.workers_per_gpu=0
# 2022-10-24 20:02:04,617 - mmpose - INFO - Epoch [1][50/18727]   lr: 4.945e-05, eta: 16 days, 8:16:20, time: 0.359, data_time: 0.198, memory: 3074, heatmap_loss: 0.0017, acc_pose: 0.0682, loss: 0.0017

# PORT=29500 CUDA_VISIBLE_DEVICES=0 ./tools/dist_train.sh configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/res50_coco_256x192.py 1 --work-dir work_dirs/res50_coco_256x192 --cfg-options data.train_dataloader.samples_per_gpu=8 data.workers_per_gpu=0
# 2022-10-24 20:59:11,341 - mmpose - INFO - Epoch [1][50/18727]   lr: 4.945e-05, eta: 10 days, 22:07:26, time: 0.240, data_time: 0.168, memory: 1418, heatmap_loss: 0.0021, acc_pose: 0.0298, loss: 0.0021

PORT=29500 CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/mobilenetv2_coco_256x192.py 4 --work-dir work_dirs/mobilenetv2_coco_256x192 --cfg-options data.train_dataloader.samples_per_gpu=32 data.workers_per_gpu=0
# 2022-10-24 21:01:16,241 - mmpose - INFO - Epoch [1][50/18727]   lr: 4.945e-05, eta: 10 days, 20:21:04, time: 0.238, data_time: 0.173, memory: 709, heatmap_loss: 0.0022, acc_pose: 0.0316, loss: 0.0022
# 2022-10-24 21:04:30,980 - mmpose - INFO - Epoch [1][50/4682]    lr: 4.945e-05, eta: 7 days, 0:44:47, time: 0.618, data_time: 0.516, memory: 2402, heatmap_loss: 0.0021, acc_pose: 0.0413, loss: 0.0021
# 2022-10-24 21:12:07,101 - mmpose - INFO - Epoch [1][50/1171]    lr: 4.945e-05, eta: 2 days, 9:09:00, time: 0.837, data_time: 0.656, memory: 2402, heatmap_loss: 0.0021, acc_pose: 0.0482, loss: 0.0021