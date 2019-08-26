import numpy as np


def dataSet():
    # 西瓜数据集离散化 简单的测试一下
    X = np.mat('2,3,3,2,1,2,3,3,3,2,1,1,2,1,3,1,2;\
            1,1,1,1,1,2,2,2,2,3,3,1,2,2,2,1,1;\
            2,3,2,3,2,2,2,2,3,1,1,2,2,3,2,2,3;\
            3,3,3,3,3,3,2,3,2,3,1,1,2,2,3,1,2;\
            1,1,1,1,1,2,2,2,2,3,3,3,1,1,2,3,2;\
            1,1,1,1,1,2,2,1,1,2,1,2,1,1,2,1,1;\
            0.697,0.774,0.634,0.668,0.556,0.403,0.481,0.437,0.666,0.243,0.245,0.343,0.639,0.657,0.360,0.593,0.719;\
            0.460,0.376,0.264,0.318,0.215,0.237,0.149,0.211,0.091,0.267,0.057,0.099,0.161,0.198,0.370,0.042,0.103\
            ').T
    X = np.array(X)  # 样本属性集合，行表示一个样本，列表示一个属性
    Y = np.mat('1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0')
    Y = np.array(Y).T  # 每个样本对应的标签
    return X,Y

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

# hideNum 对应的是hide layer 神经元的个数
# 实际在写的时候,公式出来之后,各个变量的维度要注意

def bpstand(hideNum):
    X,Y =dataSet()

    # 随机初始化权重
    V = np.random.rand(X.shape[1], hideNum)  # 权值及偏置初始化
    V_b = np.random.rand(1, hideNum)
    W = np.random.rand(hideNum, Y.shape[1])
    W_b = np.random.rand(1, Y.shape[1])


    # learning rate
    rate = 0.1

    maxTrainNum = 100000
    # stop error
    error = 0.001

    # error 以及 maxTrainNum用来限制停止条件
    trainNum = 0
    loss = 10

    while(loss > error) and (trainNum < maxTrainNum):
        for k in range(X.shape[0]):
            B = sigmoid(X[k,:].dot(V)-V_b)
            Y_ = sigmoid(B.dot(W)-W_b)
            loss = sum((Y[k]-Y_)**2)/X.shape[0]*0.5

            # 依次计算梯度
            g = Y_*(1-Y_)*(Y[k]-Y_)
            e = B*(1-B)*g.dot(W.T)
            W += rate*B.T.dot(g)
            W_b -= rate*g
            V += rate*X[k].reshape(1,X[k].size).T.dot(e)
            V_b -= rate*e
            trainNum += 1

    print("总训练次数：",trainNum)
    print("最终损失：",loss)
    print("V：", V)
    print("V_b：", V_b)
    print("W：", W)
    print("W_b：", W_b)

def bpAccum(hideNUm):
    X,Y = dataSet()

    # 随机初始化权重
    V = np.random.rand(X.shape[1], hideNum)  # 权值及偏置初始化
    V_b = np.random.rand(1, hideNum)
    W = np.random.rand(hideNum, Y.shape[1])
    W_b = np.random.rand(1, Y.shape[1])

    # learning rate
    rate = 0.1

    maxTrainNum = 100000
    # stop error
    error = 0.0001

    # error 以及 maxTrainNum用来限制停止条件
    trainNum = 0
    loss = 10

    while (loss > error) and (trainNum < maxTrainNum):
        B = sigmoid(X.dot(V) - V_b)
        Y_ = sigmoid(B.dot(W) - W_b)
        Loss = 0.5 * sum((Y - Y_) ** 2) / X.shape[0]

        # 依次计算梯度
        g = (Y - Y_) * Y_*(1 - Y_)
        e = g.dot(W.T) * B * (1 - B)
        W += rate * B.T.dot(g)
        W_b -= rate * g.sum(axis=0)
        V += rate * X.T.dot(e)
        V_b -= rate * e.sum(axis=0)
        trainNum += 1

    print("总训练次数：", trainNum)
    print("最终损失：", loss)
    print("V：", V)
    print("V_b：", V_b)
    print("W：", W)
    print("W_b：", W_b)


if __name__ == '__main__':
    bpstand(5)
    # bpAccum(5)