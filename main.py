import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class WineModel:
    def __init__(self, theta: np.array):
        self.theta = theta

    def CalculateQuality(self, atributeArr: np.array):
        return np.linalg.norm(atributeArr@self.theta)

    def norm(self, atributeArr: np.array, qualities):
        x = [[0.05]*len(qualities)]
        x = np.array(x).T
        return np.linalg.norm(atributeArr@self.theta-qualities-x,1)
        # return np.linalg.norm(atributeArr@self.theta-qualities,1)


def LoadData(csvFile):
    df = pd.read_csv(csvFile, sep=';')
    data = np.array(df.values.tolist())
    return data[:, :-1].tolist(), data[:, -1].tolist()


def LoadColumnData(csvFile, colName):
    df = pd.read_csv(csvFile, sep=';')
    return df.loc[:, colName].tolist(), df.loc[:, 'quality'].tolist()


def LoadColumnName(csvFile):
    df = pd.read_csv(csvFile, sep=';')
    return df.columns[:-1]


def getAb_All_Atributes(x, y):
    A = []
    for i,_ in enumerate(x):
        x[i].insert(0, 1)
        A.append(x[i])
    A = np.array(A)
    b = np.array(y).reshape(len(y), 1)
    return A, b


def getAb_1_Atribute(x, y):
    col1 = np.ones(len(x))
    colx = np.array(x)
    A = np.array([col1, colx]).T
    b = np.array(y).reshape(len(y), 1)
    return A, b


def getTheta(A, b):
    return np.linalg.inv(np.transpose(A)@A) @ np.transpose(A)@b


def cauA():
    csvFile = "wine.csv"
    x, y = LoadData(csvFile)
    A, b = getAb_All_Atributes(x, y)
    theta = getTheta(A, b)
    model = WineModel(theta)
    print('Norm Value: {}'.format(model.norm(A,b)))


def cauB():
    csvFile = "wine.csv"
    atributes = LoadColumnName(csvFile)
    minNorm = 1000
    bestAtribute = ''
    bestAtributeValues = []
    theta = ''
    qualities = []
    for atribute in atributes:
        x, y = LoadColumnData(csvFile, colName=atribute)
        A, b = getAb_1_Atribute(x, y)
        qualities = b
        theta = getTheta(A, b)
        norm = (A@theta-b)
        norm = np.linalg.norm(norm,1)

        if norm < minNorm:
            minNorm = norm
            bestAtribute = atribute
            bestAtributeValues = x

        print('{}   norm: {}    theta: {}'.format(atribute, norm, theta))

    print('\n\nBest Atribute\n{}: {}'.format(bestAtribute, minNorm))

    model = WineModel(theta=theta)
    plt.plot(bestAtributeValues, qualities, "o", color="blue")
    plt.title("Model base on best atribute:  {}".format(bestAtribute))
    ts = np.linspace(min(bestAtributeValues), max(bestAtributeValues))
    yts = [model.CalculateQuality([1, t]) for t in ts]
    plt.plot(ts, yts, color="red")
    plt.show()


if __name__ == '__main__':
    print('----------CAU A -----------')
    cauA()
    print('\n\n----------CAU B -----------')
    cauB()


