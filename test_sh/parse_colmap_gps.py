import numpy as np
import utils.util as util
import utils.pose_estimation_3d3d as ransacICP
import argparse
# 初始值
init_lat = 0
init_lon = 0
init_h = 0

def quaternion_conjugate(qvec):
    """计算四元数的共轭"""
    return np.array([qvec[0], -qvec[1], -qvec[2], -qvec[3]])


def quaternion_rotate_vector(qvec, vec):
    """使用四元数旋转一个向量"""
    qvec_w, qvec_x, qvec_y, qvec_z = qvec
    # 将向量表示为四元数 [0, x, y, z]
    vec_quat = np.array([0, vec[0], vec[1], vec[2]])
    # 四元数乘法  
    q_conj = quaternion_conjugate(qvec) #旋转的逆
 
    # 四元数相乘的公式 q * v * q^-1  对旋转后的结果取反
    vec_rotated = quaternion_multiply(
        quaternion_multiply(qvec, vec_quat), q_conj
    )

    # 返回旋转后的向量
    return vec_rotated[1:]


def quaternion_multiply(q1, q2):
    """四元数相乘"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2,
        w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    ])


def camera_to_world(qvec, tvec):
    """将 qvec 和 tvec 从相机坐标系转换到世界坐标系"""
    # 1. 计算四元数的共轭
    qvec_conj = quaternion_conjugate(qvec)

    # 2. 旋转 tvec 并取反
    tvec_world = -quaternion_rotate_vector(qvec_conj, tvec)

    return qvec_conj, tvec_world

def find_timestamp(filename):
    slash_index = filename.rfind('/')
    dot_index = filename.rfind('.')
    image_name = filename[slash_index + 1:dot_index]
    dir_name = filename[0:slash_index]
    return dir_name,image_name

def read_extrinsics_text(path):
    """
    参考自 https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    #  加了写文件的功能主要是为了方便调试
    dict_sampling_colmap_xyz = {} #为了防止colmap丢帧导致最后3d点对应错误，这里用map做一个映射
    video_colmap_xyz = []

    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#" \
               and (line.find("jpg")!=-1 or line.find("png")!=-1 or line.find("JPG")!=-1):
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]

                # colmap默认的是从世界到相机，此处转换 qvec 和 tvec 从相机到世界
                qvec_wc, tvec_wc = camera_to_world(qvec, tvec)

                dir_name,time_stamp = find_timestamp(image_name)

                # 保存相机在colmap世界坐标系下的位置到文件txt，格式为time_stamp x y z
                if dir_name == "110":  # 110 TODO 演示完修改 sampling_images
                    dict_sampling_colmap_xyz[time_stamp] = [tvec_wc[0],tvec_wc[1],tvec_wc[2]]
                else:
                    video_colmap_xyz.append([time_stamp,tvec_wc[0], tvec_wc[1], tvec_wc[2]])

    return dict_sampling_colmap_xyz , video_colmap_xyz

def calsrt(sampling_gps_txt):
    gps_ned = []
    colmap_ned = []

    with open(sampling_gps_txt, "r") as f1: # 读取所有镇的gps信息，
        num = 0 # 记录有多少条gps信息

        for line in f1:
            line = line.strip()
            line = line.replace("  ", " ")
            elems = line.split(" ")  # 0 图像名称（不带后缀）1-3 lat lon alt
            time_stamp = str(elems[0])
            if num==0:
                global init_lat,init_lon,init_h
                init_lat = float(elems[1])
                init_lon = float(elems[2])
                init_h = float(elems[3])

            ned_x, ned_y, ned_z = util.GPS2NED(init_lat, init_lon, init_h,
                                               float(elems[1]), float(elems[2]), float(elems[3]))
            if dict_sampling_colmap_xyz.get(time_stamp):
                gps_ned.append([ned_x,ned_y,ned_z])
                colmap_ned.append(dict_sampling_colmap_xyz.get(time_stamp))
                tmp = dict_sampling_colmap_xyz.get(time_stamp)
                # print(ned_x,' ',ned_y,' ',ned_z, "<------>",tmp[0],' ',tmp[1],' ',tmp[2])
            num+=1;
    gps_ned_np = np.array(gps_ned)

    colmap_ned_np = np.array(colmap_ned)
    # 返回 RT_34 = np.c_[sR,T]
    # pose_estimation_3d3d_ransac(t_slams, t_colmaps) # ned->slam
    return ransacICP.pose_estimation_3d3d_ransac(colmap_ned_np,gps_ned_np)

if __name__ == "__main__":

    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description='')

    # 添加参数
    parser.add_argument('--colmap_res', type=str, required=True, help='images.txt的路径')
    parser.add_argument('--sampling_gps_txt', type=str, required=True, help='采样的GPS文本文件路径')
    parser.add_argument('--vedio_gps_txt', type=str, required=True, help='视频GPS文本文件路径')

    # 解析参数
    args = parser.parse_args()

    # 使用参数
    colmap_res = args.colmap_res  # images.txt对应的文件路径
    sampling_gps_txt = args.sampling_gps_txt  # sampling图片的gps路径
    vedio_gps_txt = args.vedio_gps_txt # 输出的视频解帧的gps路径

    # 解析colmap结果
    dict_sampling_colmap_xyz , video_colmap_xyz = read_extrinsics_text(colmap_res)

    # 读入sampling图片的gps，并转化为enu坐标系进行存储，enu原点默认为第一个的gps位置
    s,R, sR, T = calsrt(sampling_gps_txt)

    # 利用sR和T计算colmap转化出来的gps
    ned_x = []
    ned_y = []
    ned_z = []
    image_name = []
    for elems in video_colmap_xyz:
        tvec = np.array(
            [[float(elems[1])], [float(elems[2])], [float(elems[3])]])
        # print(float(elems[1]), ' ', float(elems[2]), ' ', float(elems[3]))
        image_name.append(elems[0])
        # print("==================")
        # print(sR)
        # print(T)
        out_ned = sR @ tvec + T
        ned_x.append(out_ned[0][0])
        ned_y.append(out_ned[1][0])
        ned_z.append(out_ned[2][0])

    # 将gps结果写入到gps.txt文件中
    with open(vedio_gps_txt, 'w') as file:
        for i in range(len(image_name)):
            x, y, z = util.NED2GPS(init_lat, init_lon, init_h,
                                   ned_x[i], ned_y[i], ned_z[i])
            output_str = f"{image_name[i]} {x} {y} {z}\n"
            file.write(output_str)
    print("视频帧的GPS信息已写入文件：",vedio_gps_txt)