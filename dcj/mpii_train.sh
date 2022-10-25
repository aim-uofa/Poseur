# PORT=29502 CUDA_VISIBLE_DEVICES=0 ./tools/dist_train.sh configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/mpii/hourglass52_mpii_256x256.py 1 --work-dir work_dirs/hourglass52_mpii_256x256 --cfg-options data.train_dataloader.samples_per_gpu=8 data.workers_per_gpu=0
# 2022-10-25 16:25:19,127 - mmpose - INFO - Epoch [1][50/2781]    lr: 4.945e-05, eta: 4 days, 0:24:40, time: 0.594, data_time: 0.397, memory: 2840, heatmap_loss: 0.0026, acc_pose: 0.1363, loss: 0.0026

# PORT=29502 CUDA_VISIBLE_DEVICES=0 ./tools/dist_train.sh configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/mpii/res50_mpii_256x256.py 1 --work-dir work_dirs/res50_mpii_256x256 --cfg-options data.train_dataloader.samples_per_gpu=8 data.workers_per_gpu=0
# 2022-10-25 16:27:23,601 - mmpose - INFO - Epoch [1][50/2781]    lr: 4.945e-05, eta: 3 days, 1:45:43, time: 0.455, data_time: 0.373, memory: 1663, heatmap_loss: 0.0026, acc_pose: 0.0944, loss: 0.0026

# PORT=29502 CUDA_VISIBLE_DEVICES=0 ./tools/dist_train.sh configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/mpii/hrnet_w32_mpii_256x256_dark.py 1 --work-dir work_dirs/hrnet_w32_mpii_256x256_dark --cfg-options data.train_dataloader.samples_per_gpu=8 data.workers_per_gpu=0
# 2022-10-25 16:29:21,690 - mmpose - INFO - Epoch [1][50/2781]    lr: 4.945e-05, eta: 4 days, 9:32:19, time: 0.651, data_time: 0.364, memory: 2107, heatmap_loss: 0.0026, acc_pose: 0.0482, loss: 0.0026

PORT=29502 CUDA_VISIBLE_DEVICES=0 ./tools/dist_train.sh configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/mpii/litehrnet_18_mpii_256x256.py 1 --work-dir work_dirs/litehrnet_18_mpii_256x256 --cfg-options data.train_dataloader.samples_per_gpu=32 data.workers_per_gpu=0
# 2022-10-25 16:30:50,589 - mmpose - INFO - Epoch [1][50/2781]    lr: 4.945e-05, eta: 4 days, 17:53:45, time: 0.702, data_time: 0.392, memory: 1034, heatmap_loss: 0.0028, acc_pose: 0.0383, loss: 0.0028

# PORT=29502 CUDA_VISIBLE_DEVICES=0 ./tools/dist_train.sh configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/mpii/mobilenetv2_mpii_256x256.py 1 --work-dir work_dirs/mobilenetv2_mpii_256x256 --cfg-options data.train_dataloader.samples_per_gpu=8 data.workers_per_gpu=0
# 2022-10-25 16:32:07,046 - mmpose - INFO - Epoch [1][50/2781]    lr: 4.945e-05, eta: 2 days, 20:19:27, time: 0.421, data_time: 0.354, memory: 877, heatmap_loss: 0.0027, acse positive your model has flow c_pose: 0.0760, loss: 0.0027