# ############## 根据对txt文件 写入、读取数据，绘制曲线图##############
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    fp1 = open('save9/loss.txt', 'r')
    total_loss = []
    i = 0
    for loss in fp1:
        loss = loss.strip('\n')  # 将\n去掉
        loss = loss.split(' ')
        total_loss.append(loss[0])
        i += 1
    fp1.close()

    total_loss = np.array(total_loss, dtype=float)  # 将其转换成numpy的数组，并定义数据类型为float

    fp2 = open('save9/label_acc.txt', 'r')
    label_acc = []
    i = 0
    for acc in fp2:
        acc = acc.strip('\n')  # 将\n去掉
        acc = acc.split(' ')
        label_acc.append(acc[0])
        i += 1
    fp2.close()

    label_acc = np.array(label_acc, dtype=float)  # 将其转换成numpy的数组，并定义数据类型为float

    fp3 = open('save9/target_acc.txt', 'r')
    target_acc = []
    i = 0
    for acc in fp3:
        acc = acc.strip('\n')  # 将\n去掉
        acc = acc.split(' ')
        target_acc.append(acc[0])
        i += 1
    fp3.close()

    target_acc = np.array(target_acc, dtype=float)  # 将其转换成numpy的数组，并定义数据类型为float

    fp4 = open('save9/angle_acc.txt', 'r')
    angle_acc = []
    i = 0
    for acc in fp4:
        acc = acc.strip('\n')  # 将\n去掉
        acc = acc.split(' ')
        angle_acc.append(acc[0])
        i += 1
    fp4.close()

    angle_acc = np.array(angle_acc, dtype=float)  # 将其转换成numpy的数组，并定义数据类型为float

    X = np.linspace(0, i - 1, i)
    # Y1 = total_loss
    # Y2 = label_acc
    # Y3 = target_acc
    Y4 = angle_acc

    plt.figure(figsize=(8, 6))  # 定义图的大小
    plt.title("Train Result")

    plt.xlabel("Train Epoch")

    # plt.ylabel("Train Loss")
    # plt.ylabel("Label Acc")
    # plt.ylabel("Target Acc")
    plt.ylabel("Angle Acc")

    # plt.plot(X, Y1)
    # plt.plot(X, Y2)
    # plt.plot(X, Y3)
    plt.plot(X, Y4)
    plt.show()
