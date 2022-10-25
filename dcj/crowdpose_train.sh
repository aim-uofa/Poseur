# PORT=29501 CUDA_VISIBLE_DEVICES=0 ./tools/dist_train.sh configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/crowdpose/hrnet_w32_crowdpose_256x192.py 1 --work-dir work_dirs/hrnet_w32_crowdpose_256x192 --cfg-options data.train_dataloader.samples_per_gpu=4 data.workers_per_gpu=0
# 2022-10-25 16:15:36,828 - mmpose - INFO - Epoch [1][50/42733]  lr: 4.945e-05, eta: 38 days, 23:29:28, time: 0.375, data_time: 0.079, memory: 700, heatmap_loss: 0.0028, acc_pose: 0.0131, loss: 0.0028
# 2022-10-25 16:18:45,754 - mmpose - INFO - Epoch [1][50/10684]   lr: 4.945e-05, eta: 12 days, 5:33:02, time: 0.471, data_time: 0.132, memory: 1148, heatmap_loss: 0.0026, acc_pose: 0.0171, loss: 0.0026

PORT=29501 CUDA_VISIBLE_DEVICES=0 ./tools/dist_train.sh configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/crowdpose/res50_crowdpose_256x192.py 1 --work-dir work_dirs/res50_crowdpose_256x192 --cfg-options data.train_dataloader.samples_per_gpu=4 data.workers_per_gpu=0
# 2022-10-25 16:21:31,800 - mmpose - INFO - Epoch [1][50/10684]   lr: 4.945e-05, eta: 5 days, 6:01:52, time: 0.202, data_time: 0.132, memory: 1041, heatmap_loss: 0.0027, acc_pose: 0.0226, loss: 0.0027