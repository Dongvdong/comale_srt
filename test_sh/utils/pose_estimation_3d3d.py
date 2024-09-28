import math
import numpy as np
import random

def pose_estimation_3d3d_ransac(points_src, points_dst): #NED -> slam
    p = np.array(points_src, dtype=float)
    q = np.array(points_dst, dtype=float)
    print("len(points_src): ", len(points_src), " ", len(points_dst))
    # 1.计算s并去质心
    mean_p = np.mean(p, axis=0)
    mean_q = np.mean(q, axis=0)

    p_norm = p - mean_p
    q_norm = q - mean_q

    # 计算距离比
    iter_num = 0
    _s = 0
    inliner_num = 0
    while iter_num < 2000:
        # 随机挑选两个元素
        _list = []
        # print("len(points_src): ",len(points_src))
        # if len(points_src) < 2:
        #     break
        inx_1, inx_2 = random.sample(range(len(points_src)), 2)
        # print("inx_1: ",inx_1)
        # print("inx_2: ",inx_2)
        p_r = np.array([points_src[inx_1], points_src[inx_2]], dtype=float)
        q_r = np.array([points_dst[inx_1], points_dst[inx_2]], dtype=float)
        _list.append(inx_1)
        _list.append(inx_2)
        # 计算s

        p_norm_r = p_r - mean_p
        q_norm_r = q_r - mean_q

        # 所有点的xyz平方求和
        d1_list = []
        d2_list = []
        for i in range(len(q_norm_r)):
            d1 = q_norm_r[i]
            d2 = p_norm_r[i]
            dist1 = math.sqrt(np.sum(d1**2))
            dist2 = math.sqrt(np.sum(d2**2))
            d1_list.append(dist1)
            d2_list.append(dist2)
        s_r = np.sum(d1_list) / np.sum(d2_list)

        # 计算其他点s的误差值
        inliner_p = [points_src[inx_1], points_src[inx_2]]
        inliner_q = [points_dst[inx_1], points_dst[inx_2]]

        for inx in range(len(points_src)):
            # 计算点不参与验证
            if inx == inx_1 or inx == inx_2:
                continue
            p_src = np.array(points_src[inx])
            q_dst = np.array(points_dst[inx])
            # 分别计算到质心距离
            p_src_norm = p_src - mean_p
            q_dst_norm = q_dst - mean_q
            p_src_norm_dist = math.sqrt(np.sum(p_src_norm**2))
            q_dst_norm_dist = math.sqrt(np.sum(q_dst_norm**2))
            # 计算误差
            cal_dist = p_src_norm_dist * s_r
            error = cal_dist - q_dst_norm_dist
            if abs(error) < 3:
                inliner_p.append(points_src[inx])
                inliner_q.append(points_dst[inx])
                _list.append(inx)

        # 利用所有内点计算新的s
        p_r = np.array(inliner_p)
        q_r = np.array(inliner_q)
        p_norm_f = p_r - mean_p
        q_norm_f = q_r - mean_q

        d1_list = []
        d2_list = []
        for i in range(len(q_norm_f)):
            d1 = q_norm_f[i]
            d2 = p_norm_f[i]
            dist1 = math.sqrt(np.sum(d1**2))
            dist2 = math.sqrt(np.sum(d2**2))
            d1_list.append(dist1)
            d2_list.append(dist2)

        s_final = np.sum(d1_list) / np.sum(d2_list)
        # 记录内点数最高的模型
        if inliner_num < len(inliner_p):
            _s = s_final
            inliner_num = len(inliner_p)
            inx_list = _list
        iter_num += 1

    s = _s

    # 2.用s缩放src到dst尺度下
    p = s * p
    mean_p = np.mean(p, axis=0)
    p_norm = p - mean_p

    # 2.计算q1*q2^T(注意顺序：q2->q1，x是dst,y是src)
    N = len(p)

    W = np.zeros((3, 3))
    for i in range(N):
        x = q_norm[i, :]     # 每一行数据
        x = x.reshape(3, 1)  # 3行1列格式 一维数组借助reshape转置
        y = p_norm[i, :]
        y = y.reshape(1, 3)
        W += np.matmul(x, y)

    # 3.SVD分解W
    # python 线性代数库中svd求出的V与C++ Eigen库中求的V是转置关系
    U, sigma, VT = np.linalg.svd(W, full_matrices=True)
    # 旋转矩阵R
    R = np.matmul(U, VT)    # 这里无需再对V转置
    # 在寻找旋转矩阵时，有一种特殊情况需要注意。有时SVD会返回一个“反射”矩阵，这在数值上是正确的，但在现实生活中实际上是无意义的。
    # 通过检查R的行列式（来自上面的SVD）并查看它是否为负（-1）来解决。如果是，则V的第三列乘以-1。
    # 验证R行列式是否为负数   参考链接:https://blog.csdn.net/sinat_29886521/article/details/77506426
    if np.linalg.det(R) < 0:
        det = np.linalg.det(np.matmul(U, VT))
        # det 值为 -1
        mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, det]])
        ne_VT = np.matmul(VT, mat)
        R = np.matmul(U, ne_VT)
    # 平移向量
    T = mean_q - np.matmul(R, mean_p)   # dst - src
    T = T.reshape(3, 1)
    sR = s*R
    RT_34 = np.c_[sR, T]

    # 4.计算误差值
    p = np.array(points_src)
    error_sum = 0
    inx_list2 = []
    error_ENU = []
    for i in range(N):
        src = p[i, :]
        dst = q[i, :]
        src = src.reshape(3, 1)
        dst = dst.reshape(3, 1)
        test_dst = np.matmul(sR, src) + T

        error_Mat = test_dst - dst
        error_Mat2 = error_Mat**2
        error = math.sqrt(np.sum(error_Mat2))
        error_ENU.append(error)
        if error < 3:
            inx_list2.append(i)
        error_sum += error

    print("mean error:", error_sum/N)
    print("max error:", max(error_ENU))
    print("RT_34:", RT_34)
    print("R:", R)
    return s, R, sR, T