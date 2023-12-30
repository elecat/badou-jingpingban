# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
五维判断：x是一个6维向量，向量中前面一个数和后面一个数为一组，第一个数和第二个数是一组，第二个数和第三个数是一组。两个数相加，哪一组一大就数组哪一组的下标。

"""


class MultiClassModel(nn.Module):
    def __init__(self, input_size):
        super(MultiClassModel, self).__init__()
        self.linear = nn.Linear(input_size, 3)  # 线性层
        self.loss = nn.functional.cross_entropy  # loss函数采用交叉熵损失

    # 如输入真实值，返回loss值；无真实值，返回预测值
    def forward(self, x, y=None):
        y_pred = self.linear(x)  # 得到经过线性函数转换的预测值
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个6维向量，向量中前面一个数和后面一个数为一组，第一个数和第二个数是一组，第二个数和第三个数是一组。两个数相加，哪一组一大就数组哪一组的下标。
# 输出x,y。x为样本，y为结果。最大值索引。
def build_sample():
    x = np.random.random(6)
    # 组成z向量
    z1 = x[0] + x[1]
    z2 = x[2] + x[3]
    z3 = x[4] + x[5]
    z = np.array([z1, z2, z3])
    # 获取z最大值的索引
    y = np.argmax(z)
    return x, y


# 随机生成一批样本
# 正负样本均匀生成
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)


# 测试代码
# 用来测试准确率，测试的样本数量为test_sample_num，验证test_sample_num个样本的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)  # 生成测试集
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        for x_s, y_p, y_t in zip(x, y_pred, y):  # 与真实标签进行对比
            if torch.argmax(y_p) == int(y_t):
                # print("本次测试中样本")
                # print(x_s)
                # print("本次测试中预测值")
                # print(y_p)
                # print("本次测试中真实值")
                # print(y_t)
                correct += 1
            else:
                wrong += 1

    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 30  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 6  # 输入向量维度
    learning_rate = 0.001  # 学习率
    # 建立模型
    model = MultiClassModel(input_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model.pt")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    model = MultiClassficationModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        print("输入：%s, 预测类别：%s, 概率值：%s" % (vec, torch.argmax(res), res))  # 打印结果


if __name__ == "__main__":
    main()
    test_vec = [[0.47889086, 0.15229675, 0.31082123, 0.03504317, 0.18920843],
                [0.94963533, 0.5524256, 0.95758807, 0.95520434, 0.84890681],
                [0.78797868, 0.67482528, 0.13625847, 0.34675372, 0.09871392],
                [0.89349776, 0.59416669, 0.92579291, 0.41567412, 0.7358894]]
    predict("model.pt", test_vec)
