import numpy as np
from NDP_Kmeans import NDP_Kmeans
from accuracy_measures import accuracy_measures
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.preprocessing import normalize, MinMaxScaler
import os
from sklearn.manifold import TSNE

def process_data(file_path):
    data_labels = loadmat(file_path)
    keys = list(data_labels.keys())
    X = np.array(data_labels[keys[-2]], dtype=float)
    Y = np.array(data_labels[keys[-1]], dtype=int).flatten()
    return X, Y

def main():
    # root_path = './datasets/'
    # data_name = 'happy'
    # data_labels = np.loadtxt(root_path + data_name + '.txt')
    # X = data_labels[:, :-1]
    # Y = data_labels[:, -1].astype(int)
    root_path = "./RealWorld_section/"
    data_name = 'wine.mat'
    data_labels = loadmat(root_path + data_name)
    keys = list(data_labels.keys())
    X = np.array(data_labels[keys[-2]], dtype=float)
    Y = np.array(data_labels[keys[-1]], dtype=int).flatten()
    n, d = X.shape
    # 数据归一化
    X = (X - np.min(X)) / (np.max(X) - np.min(X))
    cluster_num = len(np.unique(Y))
    print('数据集名称:{}\n形状:({},{})\n聚类数:{}'.format(data_name, n, d, cluster_num))
    # NDP_Kmeans 算法
    cl = NDP_Kmeans(X, cluster_num, 0)
    # 使用 t-SNE 降维到 2D
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_2D = tsne.fit_transform(X)

    # 绘制真实标签的 t-SNE 降维可视化
    plt.figure(figsize=(8, 6))
    plt.scatter(X_2D[:, 0], X_2D[:, 1], c=Y, cmap='tab20', alpha=0.7)
    plt.title("t-SNE Visualization of Updated NDP-Kmeans Results")
    plt.colorbar()
    plt.show()

    # 绘制聚类结果的 t-SNE 降维可视化
    plt.figure(figsize=(8, 6))
    plt.scatter(X_2D[:, 0], X_2D[:, 1], c=cl, cmap='tab20', alpha=0.7)
    plt.title("t-SNE Visualization of Updated NDP-Kmeans Results")
    plt.colorbar()
    plt.show()
    statistic, ACC, PE, RE, ARI, NMI = accuracy_measures(cl, Y)
    print('ARI:{:.2f}\tNMI:{:.2f}\tACC:{:.2f}'.format(ARI, NMI, ACC))

if __name__ == "__main__":
    main()
