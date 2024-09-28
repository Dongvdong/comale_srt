import numpy as np
import utils.util as util
import utils.pose_estimation_3d3d as ransacICP
import cv2
import argparse

def quaternion_conjugate(qvec):
    """计算四元数的共轭"""
    return np.array([qvec[0], -qvec[1], -qvec[2], -qvec[3]])


def quaternion_rotate_vector(qvec, vec):
    """使用四元数旋转一个向量"""
    qvec_w, qvec_x, qvec_y, qvec_z = qvec
    # 将向量表示为四元数 [0, x, y, z]
    vec_quat = np.array([0, vec[0], vec[1], vec[2]])
    # 四元数乘法
    q_conj = quaternion_conjugate(qvec)

    # 四元数相乘的公式 q * v * q^-1
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

    dot_index = filename.rfind('.')
    image_name = filename[0:dot_index]
    
    return image_name

def read_extrinsics_text(path):
    """
    参考自 https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    dict_video_colmap_xyz = {}
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

                time_stamp = find_timestamp(image_name)

                # 保存相机在colmap世界坐标系下的位置到文件txt，格式为time_stamp x y z
                dict_video_colmap_xyz[time_stamp] = [tvec_wc[0],tvec_wc[1],tvec_wc[2]]

    return dict_video_colmap_xyz

def calsrt(video_gps_txt):
    gps_ned = []
    colmap_ned = []

    with open(video_gps_txt, "r") as f1: # 读取所有镇的gps信息，
        num = 0 # 记录有多少条gps信息

        for line in f1:
            line = line.strip()
            line = line.replace("  ", " ")
            elems = line.split(" ")  # 0 图像名称（不带后缀）1-3 lat lon alt
            time_stamp = str(elems[0])

            ned_x, ned_y, ned_z = util.GPS2NED(init_lat, init_lon, init_h,
                                               float(elems[1]), float(elems[2]), float(elems[3]))
            if dict_video_colmap_xyz.get(time_stamp):
                gps_ned.append([ned_x,ned_y,ned_z])
                colmap_ned.append(dict_video_colmap_xyz.get(time_stamp))
                tmp = dict_video_colmap_xyz.get(time_stamp)
                # print(ned_x,' ',ned_y,' ',ned_z, "<------>",time_stamp,tmp[0],' ',tmp[1],' ',tmp[2])
            num+=1
    gps_ned_np = np.array(gps_ned)

    colmap_ned_np = np.array(colmap_ned)
    # 返回 RT_34 = np.c_[sR,T]
    # pose_estimation_3d3d_ransac(t_slams, t_colmaps) # ned->slam
    return ransacICP.pose_estimation_3d3d_ransac(colmap_ned_np,gps_ned_np)

if __name__ == "__main__":
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description='处理文件路径参数')

    # 添加参数
    parser.add_argument('--colmap_res_txt', type=str, help='images.txt文件的路径')
    parser.add_argument('--gps_ned_ori', type=str, help='gps ned坐标系的原点xml文件的路径')
    parser.add_argument('--video_gps_txt', type=str, help='视频帧的gps信息文件的路径')
    parser.add_argument('--colmap2gnss_SRt_xml', type=str, help='输出的xml文件路径')

    # 解析参数
    args = parser.parse_args()

    # 使用参数
    colmap_res_txt = args.colmap_res_txt
    gps_ned_ori = args.gps_ned_ori
    video_gps_txt = args.video_gps_txt
    colmap2gnss_SRt_xml = args.colmap2gnss_SRt_xml


    # 解析colamp结果
    dict_video_colmap_xyz = read_extrinsics_text(colmap_res_txt)

    # 读入gnss ned坐标系原点
    fs = cv2.FileStorage(gps_ned_ori, cv2.FILE_STORAGE_READ)
    ori = fs.getNode('Ori').mat().astype(np.float32)
    fs.release()
    init_lat, init_lon, init_h = ori.flatten()  # 由于读取的矩阵可能是1x3，使用flatten()将其变为一维数组
    print(init_lat,' ',init_lon,' ',init_h)
    # 读入video图片的gps，计算colmap - gnss ned坐标系的SRt
    s, R, sR, T = calsrt(video_gps_txt)

    # 将 R T s分别写入xml中
    fs = cv2.FileStorage(colmap2gnss_SRt_xml, cv2.FILE_STORAGE_WRITE)
    fs.writeComment('这里的sRT是从colmap到GPS-NED坐标系下的', 0)
    fs.write('R', R)
    fs.write('T', T)
    fs.write('s', s)
    fs.release()

