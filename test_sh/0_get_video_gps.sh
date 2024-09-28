#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh #激活conda环境

# 输入
video_path="/home/yr/workspace/data/new_house4/90/90米1垂直90度.MOV" #视频路径
sampling_images_path="/home/yr/workspace/data/new_house4/110/照片"

# 输出 colmap 位姿
colmap_res_path="/home/yr/workspace/data/new_house4/90/colmap_90_gps/images.txt"

# GNSS照片抽离GNSS
sampling_images_gps_path="/home/yr/workspace/data/new_house4/110/gps_info/sampling_gps.txt"
# 视频帧算出来的gps
video_images_gps_path="/home/yr/workspace/data/new_house4/测试脚本文件夹/gps/video_gps.txt"
# 抽离的视频帧
video_images_path="/home/yr/workspace/data/new_house4/测试脚本文件夹/images"

case $1 in
    ffmpeg)
        # ffmpeg解帧 fps表示帧率，默认缩放到1920*1080
        echo "***************************开始解帧***************************"
        ffmpeg -i $video_path -vf "fps=1,scale=1920:1080" $video_images_path/%04d.jpg
        echo "===========================解帧结束============================"
        ;;
    extractGPS)
        echo "***************************提取GPS***************************"
        python extract_gps.py $sampling_images_path $sampling_images_gps_path
        echo "=========================GPS提取结束============================"        
        ;;
    colmap)
        echo "=======================这一步需要手动完成colmap========================"
        echo -e "1. 将GPS图片文件夹命名为：sampling_images\n2. 将视频帧文件夹命名为：video_images\n3. 把sampling_images和video_images统一放到一个images文件夹下面\n4. 然后开始在gui界面colmap"
        ;;
    caculateGPS)
        echo "========================计算视频帧的GPS========================"
        python  parse_colmap_gps.py --colmap_res $colmap_res_path \
                                    --sampling_gps_txt $sampling_images_gps_path \
                                    --vedio_gps_txt $video_images_gps_path
        ;;
    *)
        echo "请检查输入的参数！"
esac
