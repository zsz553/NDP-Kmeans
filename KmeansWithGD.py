import numpy as np


def KmeansWithGD(data, centers, num_clusters, graphdist, rho):
    """
    基于图距离的改进 K-Means 算法
    data: 样本数据
    centers: 初始中心（可以是索引或 'random')
    num_clusters: 聚类数目
    graphdist: 样本之间的图距离矩阵
    rho: 样本的密度值
    """
    iter = 0
    qold = np.inf
    threshold = 1e-8
    n = data.shape[0]
    if centers == 'random':
        centers = plus_init(graphdist, num_clusters, rho)
    else:
        pass  # 可以实现其他初始化方法
    while iter <= 100:
        iter += 1
        P = graphdist[centers, :]  # 计算样本到中心的距离
        ind = np.argmin(P, axis=0)  # 分配每个样本到最近的中心
        # 更新中心
        clusters = [np.where(ind == i)[0] for i in range(num_clusters)]
        for i in range(num_clusters):
            cluster_points = clusters[i]
            if len(cluster_points) == 0:
                continue  # 如果某个聚类没有样本，跳过更新
            mindist = np.inf
            mind = None
            for j in cluster_points:
                dis = np.sum(graphdist[j, cluster_points])
                if dis < mindist:
                    mindist = dis
                    mind = j
            if mind is not None:
                centers[i] = mind
        # 计算目标函数值
        P = graphdist[centers, :]
        qnew = np.sum(P[ind, range(n)])
        # 防止除以零或无穷大的情况
        if qold != 0 and not np.isinf(qold):
            delta = abs((qnew - qold) / qold)
        else:
            delta = abs(qnew - qold)
        if delta <= threshold:
            break
        qold = qnew
    cluster_labels = ind
    return cluster_labels


def plus_init(core_dist, num_clusters, rho):
    """
    改进的初始化中心点，首先选择密度最大的点作为第一个中心，
    然后选择与已有中心点距离最远的点作为下一个中心，直到选出指定数量的中心
    """
    centers = []
    maxid = np.argmax(rho)
    centers.append(maxid)
    num_centers = 1
    n = core_dist.shape[1]
    while num_centers < num_clusters:
        dists = core_dist[centers, :]
        mindist = np.min(dists, axis=0)
        maxid2 = np.argmax(mindist)
        if maxid2 in centers:
            break  # 避免重复
        centers.append(maxid2)
        num_centers += 1
    return centers
