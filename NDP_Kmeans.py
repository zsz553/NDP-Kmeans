import numpy as np
from scipy.sparse import csr_matrix
from KmeansWithGD import KmeansWithGD
from NDP_Searching import NDP_Searching
from scipy.sparse.csgraph import minimum_spanning_tree, dijkstra
import heapq
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from collections import defaultdict

class DisjointSet:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, item):
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, x, y):
        xroot = self.find(x)
        yroot = self.find(y)
        if xroot == yroot:
            return
        if self.rank[xroot] < self.rank[yroot]:
            self.parent[xroot] = yroot
        elif self.rank[xroot] > self.rank[yroot]:
            self.parent[yroot] = xroot
        else:
            self.parent[yroot] = xroot
            self.rank[xroot] += 1


def create_edge_list(weight_matrix):
    n = weight_matrix.shape[0]
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            # 每条边用 (权重, i, j) 表示
            edges.append((weight_matrix[i, j], i, j))
    return edges


def min_max(weight_matrix):
    """
    使用最小-最大路径 (path-based) 算法，将欧氏距离矩阵转换为
    “沿路径最大边最小化”的距离矩阵 K。
    K[i, j] 表示点 i 到 j 的 min-max 路径距离。
    """
    n = weight_matrix.shape[0]
    disjoint_set = DisjointSet(n)
    edges = create_edge_list(weight_matrix)
    heapq.heapify(edges)  # 将edges变成最小堆
    K = np.zeros((n, n))
    # 初始化每个点都在自己的连通分量
    components = [{i} for i in range(n)]
    while edges:
        weight, u, v = heapq.heappop(edges)
        u_root = disjoint_set.find(u)
        v_root = disjoint_set.find(v)
        # 如果这条边连接了两个不同的连通分量，就合并
        if u_root != v_root:
            disjoint_set.union(u_root, v_root)
            # 将u_root和v_root对应的components合并，并在K中更新距离
            for i in components[u_root]:
                for j in components[v_root]:
                    K[i, j] = K[j, i] = weight
            if u_root > v_root:
                u_root, v_root = v_root, u_root
            components[u_root] |= components[v_root]
            components[v_root] = components[u_root]
    np.fill_diagonal(K, 0)
    return K

def NDP_Kmeans(A, num_clusters, alpha, sigma=1.0):
    """
    input:

    基于 NDP 和改进的 K-Means (结合 path-based min_max) 进行聚类
    A: 数据集，形状为 (N, D)
    num_clusters: 聚类数目
    alpha: 用于过滤低密度核心点的阈值比例

    return:

    index: 每个数据点最近邻按距离排序后的索引矩阵
    supk: 最终使用的K值,即自然最近邻居的平均数量
    nb: 每个数据点作为其他数据点的自然邻居的次数
    rho: 每个数据点的局部密度
    local_core: 每个数据点对应的局部核心点
    cores: 所有准核心点的索引列表
    cl: 每个数据点所属的簇编号
    cluster_number: 检测到的簇总数
    """
    N, dim = A.shape
    # 1) 搜索自然密度峰（NDP）
    index, supk, nb, rho, local_core, cores, cl, cluster_number = NDP_Searching(A)

    # 2) 根据 alpha 排除低密度核心点
    rho_sorted = np.sort(rho)
    if alpha == 0:
        rho_threshold = 0
    else:
        rho_threshold = rho_sorted[int(N * alpha)]

    cores = cores.tolist()
    for i in range(cluster_number):
        if cores[i] != 0 and rho[cores[i]] < rho_threshold:
            mind = np.inf
            p = -1
            for j in range(cluster_number):
                if i != j and rho[cores[j]] > rho_threshold:
                    distance = np.linalg.norm(A[cores[i], :] - A[cores[j], :])
                    if distance < mind:
                        mind = distance
                        p = j
            for j in range(N):
                if local_core[j] == cores[i]:
                    local_core[j] = cores[p]

    # 3) 重新确定核心点 (去除低密度后)
    cluster_number = 0
    cl = np.zeros(N, dtype=int)
    # 用于存储准核心的数组：存储的是原始数据中的index
    cores2 = []
    for i in range(N):
        if local_core[i] == i:
            cores2.append(i)
            cluster_number += 1
            cl[i] = cluster_number - 1
    cores2 = np.array(cores2)
    core_to_cluster = {core: idx for idx, core in enumerate(cores2)}
    for i in range(N):
        cl[i] = core_to_cluster[local_core[i]]
    # --------------若核心点数 < 需要聚类数，则提前返回--------------
    if cluster_number < num_clusters:
        print(
            f"[WARNING] Number of core points ({cluster_number}) "
            f"is less than requested clusters ({num_clusters}). "
            "Returning the preliminary assignment (cl)."
        )
        return cl
    # 4) 计算核心点之间的max_d（欧式距离）
    max_eu_dist = 0.0
    for i in range(cluster_number):
        for j in range(i + 1, cluster_number):
            d_eu = np.linalg.norm(A[cores2[i], :] - A[cores2[j], :])
            if d_eu > max_eu_dist:
                max_eu_dist = d_eu
    # 5）计算每个子簇的MNDP（每个子簇元素组成）
    MNDP = defaultdict(list)
    for i, value in enumerate(local_core):
        MNDP[value].append(i)
    # 6）计算每个子簇的NNDP（每个子簇的邻域）
    NNDP = defaultdict(list)
    for i, value in enumerate(local_core):
        NNDP[value].append(index[i, :supk])
    # 使用字典推导式合并数组
    NNDP = {value: np.concatenate(arrays) for value, arrays in NNDP.items()}
    # 7）计算基于共享近邻的子簇距离ND(p,q)
    core_dist = np.zeros((cluster_number, cluster_number))
    for i in range(cluster_number):
        for j in range(i + 1, cluster_number):
            p = cores2[i]
            q = cores2[j]
            # Euclidean距离 d(p,q)
            d_eu = np.linalg.norm(A[p, :] - A[q, :])
            # CN(p,q) = p、q的共享近邻
            neighbors_p = set(NNDP[p])
            neighbors_q = set(NNDP[q])
            cn = neighbors_p.intersection(neighbors_q)
            if len(cn) > 0:
                # 若 |CN(p,q)| > 0
                sum_naden = np.sum(rho[list(cn)])  # NaDen(o) = rho[o]
                core_dist[i, j] = d_eu / (len(cn) * sum_naden)
            else:
                # 若 |CN(p,q)| = 0
                core_dist[i, j] = max_eu_dist * (1.0 + d_eu)
            # 对称赋值
            core_dist[j, i] = core_dist[i, j]
    # 9) 用 min_max 做 path-based距离。
    path_dist = min_max(core_dist)
    # 将 path_dist 转化为相似度矩阵 W = exp(- path_dist^2 / (2*sigma^2))
    W = np.exp(-(path_dist**2) / (2 * (sigma**2)))
    # 10) 做谱聚类(未归一化拉普拉斯 + 特征分解 + KMeans)
    #   a) 构造拉普拉斯矩阵 L = D - W
    D = np.diag(W.sum(axis=1))
    D_inv_sqrt = np.diag(1 / np.sqrt(W.sum(axis=1)))
    L = D - W
    #   b) 特征分解，取前 num_clusters 个最小特征值对应向量
    eigvals, eigvects = np.linalg.eigh(D_inv_sqrt @ L @ D_inv_sqrt)
    idx = np.argsort(eigvals)
    X_spec = eigvects[:, idx[:num_clusters]]
    #   c) 行归一化
    X_spec = normalize(X_spec, norm="l2", axis=1)
    #   d) 在谱特征空间做 KMeans
    kmeans = KMeans(n_clusters=num_clusters, n_init=40)
    core_cl_labels = kmeans.fit_predict(X_spec)
    # 11) 映射回所有样本
    cl2 = np.zeros(N, dtype=int)
    for idxc, c_id in enumerate(cores2):
        cl2[c_id] = core_cl_labels[idxc]
    for i in range(N):
        cl2[i] = cl2[local_core[i]]
    return cl2