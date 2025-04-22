import numpy as np
from sklearn.neighbors import NearestNeighbors


def NDP_Searching(A):
    """
    搜索自然密度峰(NDP)
    """
    N, dim = A.shape
    max_neighbors = min(20000, N)
    nbrs = NearestNeighbors(n_neighbors=max_neighbors, algorithm='kd_tree').fit(A)
    distances, indices = nbrs.kneighbors(A)
    index = indices  # index[i, :] 表示第 i 个样本的邻居索引
    # 初始化基本数据
    r = 1
    flag = 0
    nb = np.zeros(N, dtype=int)
    count = 0
    count1 = 0
    count2 = 0
    # 搜索自然最近邻居
    while flag == 0:
        if r >= index.shape[1]:
            print("已达到最大邻居数")
            break
        for i in range(N):
            k = index[i, r]
            nb[k] += 1
        r += 1
        count2 = np.sum(nb == 0)
        if count1 == count2:
            count += 1
        else:
            count = 1
        if count2 == 0 or (r > 2 and count >= 2):
            flag = 1
        count1 = count2
    supk = r - 1
    max_nb = np.max(nb)
    rho = np.zeros(N)
    Non = max_nb
    # 计算密度 rho
    for i in range(N):
        d = 0
        for j in range(Non + 1):
            if j >= index.shape[1]:
                break
            x = index[i, j]
            d += np.linalg.norm(A[i, :] - A[x, :])
        rho[i] = Non / d if d != 0 else 0
    # 计算局部核心点
    local_core = np.zeros(N, dtype=int)
    for i in range(N):
        rep = i
        xrho = rho[rep]
        for j in range(1, supk + 1):
            if j >= index.shape[1]:
                break
            if xrho < rho[index[i, j]]:
                xrho = rho[index[i, j]]
                rep = index[i, j]
        local_core[i] = rep
    # 根据 UR 更新 local_core
    visited = np.zeros(N, dtype=int)
    for k in range(N):
        if visited[k] == 0:
            parent = k
            path = []
            while local_core[parent] != parent:
                visited[parent] = 1
                path.append(parent)
                parent = local_core[parent]
            for node in path:
                local_core[node] = parent
    # 获取 NDPs
    cluster_number = 0
    cl = np.zeros(N, dtype=int)
    cores = []
    for i in range(N):
        if local_core[i] == i:
            cores.append(i)
            cl[i] = cluster_number
            cluster_number += 1
    cores = np.array(cores)
    core_to_cluster = {core: idx for idx, core in enumerate(cores)}
    for i in range(N):
        cl[i] = core_to_cluster[local_core[i]]
    return index, supk, nb, rho, local_core, cores, cl, cluster_number