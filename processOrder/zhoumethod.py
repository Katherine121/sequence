import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def find_best_k(path):
    """
    find best number of clusters.
    :param path: dataset path.
    :return:
    """

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

            lat_index = file.find("lat")
            alt_index = file.find("alt")
            lon_index = file.find("lon")

            start = file[0: lat_index - 1]
            lat_pos = file[lat_index + 4: alt_index - 1]
            alt_pos = file[alt_index + 4: lon_index - 1]
            lon_pos = file[lon_index + 4: -4]

            labels.append(list(map(eval, [lat_pos, lon_pos])))

    featureList = ['0', '1']
    mdl = pd.DataFrame.from_records(labels, columns=featureList)

    SSE = []
    for i in range(60, 200, 20):
        print(i)
        estimator = KMeans(n_clusters=i)
        estimator.fit(np.array(mdl[['0', '1']]))
        SSE.append(estimator.inertia_)
    X = range(60, 200, 20)
    plt.xlabel('i')
    plt.ylabel('SSE')
    plt.plot(X, SSE, 'o-')
    plt.show()


if __name__ == "__main__":
    find_best_k(path="order")
