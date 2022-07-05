import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.fftpack import fft, ifft
import pickle
from tsfresh import extract_relevant_features, extract_features

from pylab import mpl
from sklearn.cluster import KMeans, DBSCAN, kmeans_plusplus
from sklearn.preprocessing import label_binarize, StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.metrics import roc_curve, auc, confusion_matrix, roc_auc_score, \
    silhouette_samples, silhouette_score, precision_recall_curve, average_precision_score, f1_score, \
    calinski_harabasz_score
from sklearn.cluster import AgglomerativeClustering

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题




# 快速傅里叶变换
def get_fft_power_spectrum(y_values, N, f_s, f=2):
    """
    :param y_values:原始信号数据
    :param N:数据采样点数
    :param f_s:采样频率
    :param f:双边谱与单边谱换算，所以要除以2
    :return: f_values设置的范围，fft_values为所有信号点的傅里叶变换值，ps_values是直接周期法功率, ps_cor_values是自相关下的对数功率
    
    """
    f_values = np.linspace(0.0, f_s / f, N / f)
    fft_values_ = np.abs(fft(y_values))
    fft_values = 2.0 / N * (fft_values_[0:N / 2])  # 频率真实幅值分布，单边频谱图，再把所有点频率大小表示出来*2
    
    # power spectrum 直接周期法
    ps_values = fft_values ** 2 / N
    
    # 自相关傅里叶变换法
    cor_x = np.correlate(y_values, y_values, 'same')  # 自相关
    cor_X = fft(cor_x, N)
    ps_cor = np.abs(cor_X)
    ps_cor_values = 10 * np.log10(ps_cor[0:N / 2] / np.max(ps_cor))
    
    return f_values, fft_values, ps_values, ps_cor_values


# 计算时域特征
def psfeatureTime(data, p1, p2):
    """
    :param data:原始信号数据
    :param p1:计算起始点
    :param p2:计算终止点
    :return: 返回10个时域特征
    """
    # 均值
    df_mean = data.iloc[:, p1:p2].mean(axis=1)
    # 方差
    df_var = data.iloc[:, p1:p2].var(axis=1)
    # 标准差
    df_std = data.iloc[:, p1:p2].std(axis=1)
    # 均方根
    df_rms = (pow(df_mean, 2) + pow(df_std, 2)).apply(lambda x: math.sqrt(x))
    # df_rms=math.sqrt(pow(df_mean,2) + pow(df_std,2))
    # 偏度
    df_skew = data.iloc[:, p1:p2].skew(axis=1)
    # 峭度
    df_kurt = data.iloc[:, p1:p2].kurt(axis=1)
    sum = 0
    for i in range(p1, p2):
        sum += abs(data.iloc[:, i]).apply(lambda x: math.sqrt(x))
    # 波形因子
    df_boxing = df_rms / (abs(data.iloc[:, p1:p2]).mean(axis=1))
    # 峰值因子
    df_fengzhi = (max(data.iloc[:, p1:p2])) / df_rms
    # 脉冲因子
    df_maichong = (max(data.iloc[:, p1:p2])) / (abs(data.iloc[:, p1:p2]).mean(axis=1))
    # 裕度因子
    df_yudu = (max(data.iloc[:, p1:p2])) / pow((sum / (p2 - p1)), 2)
    
    # 合并特征并输出
    featuretime_list = pd.DataFrame(
        list(zip(df_mean, df_var, df_std, df_rms, df_skew, df_kurt, df_boxing, df_fengzhi, df_maichong, df_yudu)))
    featuretime_list.columns = ['均值', '方差', '标准差', '均方根', '偏度', '峭度', '波形因子', '峰值因子', '脉冲因子', '裕度因子']
    return featuretime_list


# 计算频域特征
def psfeatureFre(data, p1, p2, f_s=12000):
    """
    :param data: 原始信号数据
    :param p1: 计算起始点
    :param p2: 计算终止点
    :param f_s: 信号频率
    :return: 4个频域特征
    """
    
    # ----------重心频率------------
    N = p2 - p1
    f_values, fft_values, ps_values, ps_cor_values = get_fft_power_spectrum(data.iloc[:, p1:p2], N, f_s, 2)
    # 直接取周期法功率
    P = ps_values
    
    S = []
    for i in range(N // 2):
        P1 = P[i]
        f1 = fft_values[i]
        s1 = P1 * f1
        S.append(s1)
    
    # 求取重心频率
    S1 = np.sum(S) / np.sum(P)
    
    # -----------平均频率------------
    S2 = np.sum(P) / N
    
    # -----------频率标准差------------
    S = []
    for i in range(N // 2):
        P1 = P[i]
        f1 = fft_values[i]
        s2 = P1 * ((f1 - S1) ** 2)
        S.append(s2)
    
    S3 = np.sqrt(np.sum(S) / np.sum(P))
    # -----------均方根频率-----------
    S = []
    for i in range(N // 2):
        P1 = P[i]
        f1 = fft_values[i]
        s2 = P1 * (f1 ** 2)
        S.append(s2)
    
    S4 = np.sqrt(np.sum(S) / np.sum(P))
    
    featureRre_list = pd.DataFrame(list(zip(S1, S2, S3, S4)))
    featureRre_list.columns = ['重心频率', '平均频率', '频率标准差', '均方根频率']
    
    return featureRre_list


# 数据分箱
def mono_bin(data):
    d_out = pd.DataFrame()
    for i in range(data.shape[1]):
        d_temp = (pd.cut(data.iloc[:, i], 3, labels=['高', '中', '低']))
        d_out = pd.concat([d_out, d_temp.to_frame()], axis=1)
    d_out.columns = data.columns
    return d_out


# 聚类方法
class Clustering:
    
    # K-means聚类的评估，以便于找到最佳聚类数
    # inertia和轮廓系数的对比，评估聚类效果
    def Kmeans_assess(self, data):
        X = []
        inertia_scores = []
        km_scores = []  # 轮廓系数
        km_scores2 = []  # CH指数
        
        # 遍历n选择合适的聚类数
        for n in range(2, 20):
            km = KMeans(n_clusters=n).fit(data.values)
            # 轮廓系数接收的参数中，第二个参数至少有两个分类
            sc = silhouette_score(data.values, km.labels_)
            km_scores.append(sc)
            inertia_scores.append(km.inertia_)
            
            sc2 = calinski_harabasz_score(data.values, km.labels_)
            km_scores2.append(sc2)
            X.append(n)
        
        # 可视化聚类评估曲线，进行聚类数n的选择
        plt.plot(X, inertia_scores, label='SSE', color='#4286F3')
        plt.xlabel('聚类数')
        plt.ylabel('得分')
        plt.legend()
        plt.show()
        
        plt.subplot(211)
        plt.plot(X, km_scores, label='轮廓系数', color='#4286F3')
        plt.ylabel('得分')
        plt.legend()
        
        plt.subplot(212)
        plt.plot(X, km_scores2, label='CH指数', color='#4286F3')
        plt.xlabel('聚类数')
        plt.ylabel('得分')
        plt.legend()
        plt.show()


# 数据读取
data = pd.read_csv('原始数据//zhenjiang_power.csv')
data['record_date'] = pd.to_datetime(data['record_date'])
uesrId = set(data['user_id'])

# 构造聚类数据集
data_train = pd.DataFrame()
for i, name in enumerate(uesrId):
    data_temp = data[data['user_id'] == name]
    data_temp_x = np.array(data_temp['power_consumption']).reshape((1, -1))
    data_temp_x = pd.DataFrame(data_temp_x)
    data_temp_x['user_id'] = name
    data_train = data_train.append(data_temp_x)
data_train = data_train.reset_index(drop=True)

# 标准化
min_max_scaler = MinMaxScaler()
std = StandardScaler()

# 删除特殊数据，不删除会导致聚类有误？类别比例失衡
data_train = data_train.drop(index=[173, 174, 1415, 89, 128, 1259, 1304, 1307])

# 构造时域特征
time_feature = psfeatureTime(data_train.iloc[:, :-1], 0, 600)
time_feature = pd.concat([time_feature, data_train.iloc[:, -1].to_frame().reset_index(drop=True)], axis=1)

# 时域特征归一化
time_feature_minMax = min_max_scaler.fit_transform(time_feature.iloc[:, :-1])

# 时域特征数据分箱
time_feature_minMax_bin = mono_bin(pd.DataFrame(time_feature_minMax))

# 独热编码
gen_ohe = OneHotEncoder()
gen_feature_arr = gen_ohe.fit_transform(time_feature_minMax_bin).toarray()
gen_feature = pd.DataFrame(gen_feature_arr)

# 特征热力图
sns.heatmap(gen_feature.corr())
plt.show()

# 构造频域特征,暂时还未进行


# 聚类效果评估部分
cluster = Clustering()
cluster.Kmeans_assess(data=gen_feature)

# 确定聚类数后进行Kmeans聚类，并进行可视化
n_clu = 5  # 聚类数为5
km = KMeans(n_clusters=n_clu).fit(gen_feature.values)
gen_feature['labels'] = km.labels_  # 新建标签列

# 不同类别比例的饼图绘制
gen_feature['labels'].value_counts().plot.pie()
plt.show()

# 观察其余类别数据
Id = gen_feature[gen_feature['labels'].isin([1])]
print(Id.index)

# 可视化类别
fig1 = plt.figure(1)
for i in range(n_clu):
    plt.subplot(510 + i + 1)
    gen_feature[gen_feature['labels'] == i].iloc[0, :-1].plot()
plt.title('聚类后各类别编码展示')
fig1.show()

# 可视化聚类中心
fig2 = plt.figure(2)
for i in range(n_clu):
    plt.subplot(510 + i + 1)
    plt.plot(km.cluster_centers_[i])
plt.title('聚类中心编码展示')
fig2.show()

plt.show()
