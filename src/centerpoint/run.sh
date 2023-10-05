./build/main --model_file /home/nvidia/way/Ted-Paddle3D/output/ted.pdmodel  \
--params_file /home/nvidia/way/Ted-Paddle3D/output/ted.pdiparams \
--lidar_file /home/nvidia/way/OpenPCDet/data/kitti/training/velodyne/000006.bin \
--num_point_dim 4 \
--point_cloud_range "0 -40 -3 70.4 40 1" \
--use_trt 0 --collect_shape_info 0 \
--dynamic_shape_file /home/nvidia/way/Ted-Paddle3D/output/shape_info.txt \
--trt_static_dir /home/nvidia/way/Ted-Paddle3D/output/
