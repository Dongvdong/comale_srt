#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh #激活conda环境

colmap_res_txt="/home/r9000k/v2_project/data/NWPU/sparse/0/images.txt" #图像colmap重建位姿结果
gps_ned_ori="ori.xml" #图像gnss参考点
video_gps_txt="/home/r9000k/v2_project/data/NWPU/FHY_config/FHY_gps.txt" #图像GNSS数据
colmap2gnss_SRt_xml="srt.xml" #计算的srt关系

# pip install geographiclib
conda activate gaussian_splatting
python parse_colmap_map.py --colmap_res_txt $colmap_res_txt \
                          --gps_ned_ori $gps_ned_ori \
                          --video_gps_txt $video_gps_txt \
                          --colmap2gnss_SRt_xml $colmap2gnss_SRt_xml