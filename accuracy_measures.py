import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment


def accuracy_measures(new_class, true_class):
    """
    计算聚类算法的性能指标
    """
    new_class = np.array(new_class).astype(int)
    true_class = np.array(true_class).astype(int)
    # 调整标签，使其从0开始
    new_class -= new_class.min()
    true_class -= true_class.min()
    sums, cluster = equation(new_class)
    r = len(new_class)
    l = len(sums)
    cs = np.unique(true_class)
    n = len(cs) + 2
    statistic = np.zeros((l, n))
    statistic[:, 0] = sums
    ms = 0
    PE = 0
    RE = 0
    true_class_indicators = np.zeros(r)
    for i in range(l):
        indices = cluster[i]
        s = class_distribution(indices, cs, true_class)
        statistic[i, 1:n-1] = s
        maxvalue = s.max()
        maxrow = s.argmax()
        attr_class = np.unique(true_class)
        for idx in indices:
            if true_class[idx] == attr_class[maxrow]:
                true_class_indicators[idx] = 1
        statistic[i, n-1] = maxvalue / sums[i]
        PE += maxvalue / sums[i]
        ms += maxvalue
    for i in range(l):
        maxvalue = statistic[i, 1:n-1].max()
        maxrow = statistic[i, 1:n-1].argmax()
        if statistic[:, maxrow + 1].sum() != 0:
            RE += maxvalue / statistic[:, maxrow + 1].sum()
    PE = PE / l
    RE = RE / l
    ACC = cluster_accuracy(new_class, true_class) * 100
    ARI = adjusted_rand_score(true_class, new_class) * 100
    NMI = normalized_mutual_info_score(true_class, new_class) * 100
    return statistic, ACC, PE, RE, ARI, NMI

def equation(labels):
    """
    计算每个聚类的大小和包含的样本索引
    """
    k = np.unique(labels)
    m = len(k)
    sums = np.zeros(m, dtype=int)
    cluster = [[] for _ in range(m)]
    label_to_index = {label: idx for idx, label in enumerate(k)}
    for idx, label in enumerate(labels):
        cluster_idx = label_to_index[label]
        sums[cluster_idx] += 1
        cluster[cluster_idx].append(idx)
    return sums, cluster


def class_distribution(indices, attribute, data):
    """
    计算每个聚类中不同真实类别的样本数量
    """
    n = len(indices)
    k = len(attribute)
    sum_ = np.zeros(k, dtype=int)
    attribute_to_index = {attr: idx for idx, attr in enumerate(attribute)}
    for idx in indices:
        label = data[idx]
        j = attribute_to_index[label]
        sum_[j] += 1
    return sum_


def cluster_accuracy(dataCluster, dataLabel):
    """
    计算聚类准确率
    """
    dataCluster = np.array(dataCluster).astype(int)
    dataLabel = np.array(dataLabel).astype(int)
    nData = len(dataLabel)
    nC1 = dataCluster.max() + 1
    nC2 = dataLabel.max() + 1
    E = np.zeros((nC1, nC2), dtype=int)
    for m in range(nData):
        i1 = dataCluster[m]
        i2 = dataLabel[m]
        E[i1, i2] += 1
    cost_matrix = -E
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    nMatch = -cost_matrix[row_ind, col_ind].sum()
    ACC = nMatch / nData
    return ACC