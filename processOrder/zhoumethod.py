import os
import pandas as pd  # 导入pandas库
import numpy as np
from sklearn.cluster import KMeans  # 导入k均值函数
import matplotlib.pyplot as plt


path = "./order"

pics_list = []
labels = []

dir_path = os.listdir(path)
dir_path.sort()

for dir in dir_path:
    full_dir_path = os.path.join(path, dir)

    file_path = os.listdir(full_dir_path)
    file_path.sort()

    for file in file_path:
        full_file_path = os.path.join(full_dir_path, file)

        pics_list.append(full_file_path)

        # 纬度
        lat_index = file.find("lat")
        # 高度
        alt_index = file.find("alt")
        # 经度
        lon_index = file.find("lon")

        start = file[0: lat_index - 1]
        lat_pos = file[lat_index + 4: alt_index - 1]
        alt_pos = file[alt_index + 4: lon_index - 1]
        lon_pos = file[lon_index + 4: -4]

        labels.append(list(map(eval, [lat_pos, lon_pos])))

featureList = ['0', '1']  # 创建了一个特征列表，这是我原始表格里的特征名
mdl = pd.DataFrame.from_records(labels, columns=featureList)    # 把labels里的数据放进来，列的名称=featurelist

# '利用SSE选择k'
SSE = []  # 存放每次结果的误差平方和
for i in range(60, 200, 20):  # 尝试要聚成的类数
    print(i)
    estimator = KMeans(n_clusters=i)  # 构造聚类器
    estimator.fit(np.array(mdl[['0', '1']]))
    SSE.append(estimator.inertia_)
X = range(60, 200, 20)  # 跟k值要一样
plt.xlabel('i')
plt.ylabel('SSE')
plt.plot(X, SSE, 'o-')
plt.show()  # 画出图
