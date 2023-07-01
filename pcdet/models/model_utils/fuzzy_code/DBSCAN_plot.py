# -*- coding: utf-8 -*-
# @Time    : 2021/9/24 23:26
# @Author  : ZFC
# @File    : DBSCAN_plot.py
# @Task    :
from cProfile import label
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import sys
import numpy as np

def DBSCAN_plot(X):
    x= StandardScaler().fit_transform(X)#数据标准化,归一化
    #设置DBSCAN
    db = DBSCAN(eps=0.3, min_samples=5).fit(x)
    #eps:即我们的ϵ-邻域的距离阈值，和样本距离超过ϵ的样本点不在ϵ-邻域内。默认值是0.5
    #min_samples：即样本点要成为核心对象所需要的ϵ-邻域的样本数阈值。默认值是5.
    labels = db.labels_#聚类标签，噪声样本标签为-1
    #print("框的数量",labels.shape)

    unique, counts = np.unique(labels, return_counts=True)#对labels进行统计，unique：标签, counts：统计结果
    #print("统计结果：",dict(zip(unique, counts)))#查看统计情况

    #创建次数统计结果，对于噪声默认为4，因为DBSCAN的样本阈值数为5
    density_count=np.ones(len(labels))*4#生成全为4的矩阵，是一个行,长度为len(labels)，代表密度
    for i,label in enumerate(unique):#返回下标和标签
        if label==-1:
            continue
        cls_mask = (labels == label)  #获取非噪声标签的坐标
        density_count[cls_mask]=counts[i]  #修改标签对应的结果
    #print("密度统计结果",density_count)

    #获取最大、最小统计值
    if unique.size >0:#当有聚类时，才执行下面的内容
        if -1 in unique:#当有噪声时，去除噪声值
            if counts[1:].size == 0:
                max_count=100#设置聚类的最低点数
            else:
                max_count=max(counts[1:])#去除噪声后，从中选取最大的
        else:
            max_count = max(counts)

    #归一化,(使其不严格为0，不严格为1)
    fac = 0.99 /max_count
    density= density_count* fac
    #print("密度",density)
    
    # with open('density.csv','ab') as f:
    #     np.savetxt(f, density, delimiter=',')#将数据保存成csv，保存在服务器...OpenPCDet/tools

    # plot_DB(X,labels,mode="save",path="DBSCAN")#绘制聚类图像
    return density


def plot_DB(X,labels,mode="plot",path="none"):#输入为原始坐标和标签,mode="plot"是绘制图像，"save"是保存数据
    #将数据保存到csv中，不再绘制
    if mode=="save":
        labels=labels.reshape(-1,1)#行转成列
        merged=np.hstack((X,labels))#按列拼接
        with open(path+'.csv','ab') as f:
            np.savetxt(f, merged, delimiter=',')#将数据保存成csv，保存在服务器...OpenPCDet/tools

    # 绘制结果
    if mode=="plot":
        fig = plt.figure(dpi=128, figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        #绘制噪声的坐标
        if -1 in labels:#有噪声时，噪声和非噪声分别绘制
            class_noice_mask = (labels == -1)#获取噪声下标
            xyz_noice = X[class_noice_mask]#获取噪声的原始坐标
            ax.scatter(xyz_noice[:,0], xyz_noice[:,1],xyz_noice[:,2], c = 'k')#绘制噪声坐标，颜色为黑色

            #绘制非噪声坐标
            class_mask=~class_noice_mask #取反，获取非噪声下标
            xyz = X[class_mask]  # 获取非噪声的原始坐标
            labels=labels[class_mask]#获取非噪声的标签
            ax.scatter(xyz[:,0], xyz[:,1],xyz[:,2], c=labels, cmap=plt.cm.Spectral)
            plt.tick_params(labelsize=26)

            plt.title('DBSCAN result',fontsize=26)
            plt.savefig('DBSCAN result.png')
            # plt.show()
        else:#无噪声时，非噪声直接绘制
            xyz = X  #获取非噪声的原始坐标
            ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=labels, cmap=plt.cm.Spectral)
            plt.tick_params(labelsize=26)

            plt.title('FUZZY result',fontsize=26)
            plt.savefig('FUZZY result.png')
            # plt.show()
    # sys.exit(0)  # 正常退出程序




#获取for循环获取距离，以每个点为中心设置方框
#data只有（x,y,z）
def cal_density(data_xyz,width=5):
    density=np.zeros(len(data_xyz))
    for i in range(len(data_xyz)):#对每个点循环
        count=0
        xmin=data_xyz[i][0]-width
        xmax = data_xyz[i][0] + width
        ymin = data_xyz[i][1] - width
        ymax = data_xyz[i][1] + width
        zmin = data_xyz[i][2] - width
        zmax = data_xyz[i][2] + width
        for j in range(len(data_xyz)):#对每个点循环
            if j!=i:
                if data_xyz[j][0]>=xmin and data_xyz[j][0]<=xmax:
                    if data_xyz[j][1] >= ymin and data_xyz[j][1] <= ymax:
                        if data_xyz[j][2] >= zmin and data_xyz[j][2] <= zmax:
                            count+=1
                        else:
                            continue
        density[i]=count
    return density