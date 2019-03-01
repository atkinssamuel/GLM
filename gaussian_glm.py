
from data_utils import load_dataset
from svd_glm import sinFt, polyFt, expFt, RMSE, getPhi
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import math
from operator import itemgetter


x_train, x_valid, x_test, y_train, y_valid, y_test = ([[],[],[],[],[]], [[],[],[],[],[]],
                                                     [[],[],[],[],[]], [[],[],[],[],[]],
                                                     [[],[],[],[],[]], [[],[],[],[],[]])
x_train[0], x_valid[0], x_test[0], y_train[0], y_valid[0], y_test[0] = list(load_dataset('mauna_loa'))
x_train[1], x_valid[1], x_test[1], y_train[1], y_valid[1], y_test[1] = list(load_dataset('rosenbrock', n_train=1000, d=2))
x_train[2], x_valid[2], x_test[2], y_train[2], y_valid[2], y_test[2] = list(load_dataset('pumadyn32nm'))
x_train[3], x_valid[3], x_test[3], y_train[3], y_valid[3], y_test[3] = list(load_dataset('iris'))
x_train[4], x_valid[4], x_test[4], y_train[4], y_valid[4], y_test[4] = list(load_dataset('mnist_small'))
inf = 10000000

def intClassification(data):
    converted = []
    for element in data:
        converted = np.append(converted, np.where(element == True))
    return converted

def incorrectPercentage(a, b):
    a = list(a)
    b = list(b)
    incorrectCount = 0
    for i in range(len(a)):
        if(list(a[i]) != list(b[i])):
            incorrectCount += 1
    return incorrectCount/len(a)

def performancePlotGaussianGLM(x_train_valid, y_train_valid, train_valid_predictions, xTest, yTest, predictions):
    fig, ax = plt.subplots()
    ax.plot(x_train_valid, y_train_valid, 'yo', markersize = 1, label='Training and Validation Data')
    ax.plot(x_train_valid, train_valid_predictions, 'bo', markersize = 0.5, label='Training and Validation Prediction')
    ax.plot(xTest, yTest, 'bo', markersize = 1, label='Test Data')
    ax.plot(xTest, predictions, 'mo', markersize = 0.5, label='Test Prediction')
    ax.legend()
    plt.xlabel('x Values')
    plt.ylabel('y Values')
    plt.title('Training and Testing Actuals and Predictions for Gaussian GLM')
    fig.savefig('results/GaussianGLMTrainingTestingPred.png')
    plt.show()
    return

def createGaussianK(x, z, theta):
    # If the dataset is one dimensional, we can compute k quickly using the following:
    if np.shape(x)[0] == 1:
        predMatrix = np.exp(-np.square(x-np.transpose(z))/theta)
        return predMatrix
    predMatrix = np.ones((1000, 1))
    x = np.expand_dims(x, axis=1)
    firstValueFlag = True
    for zValue in z:
        zValue = np.transpose(np.expand_dims(zValue, axis = 1))
        predMatrixColumn = np.exp(-np.sum(np.square(x - zValue), axis = 2)/theta)
        if firstValueFlag:
            firstValueFlag = False
            predMatrix = predMatrixColumn
        else:
            predMatrix = np.append(predMatrix, predMatrixColumn, axis = 1)
    return predMatrix

def gaussian_glm_regression(regressInd):
    xTrainValid = np.append(x_train[regressInd], x_valid[regressInd], axis = 0)
    yTrainValid = np.append(y_train[regressInd], y_valid[regressInd], axis = 0)
    minL, minS = findParameters(regressInd)
    predMatrixTrain = createGaussianK(xTrainValid, xTrainValid, minS)
    predMatrixTest = createGaussianK(x_test[regressInd], xTrainValid, minS)
    cholesky = sp.linalg.cho_factor(predMatrixTrain + minL * np.identity(len(predMatrixTrain)))
    alpha = sp.linalg.cho_solve(cholesky, yTrainValid)
    trainPred = np.transpose(np.matmul(np.transpose(alpha), np.transpose(predMatrixTrain)))
    testPred = np.transpose(np.matmul(np.transpose(alpha), np.transpose(predMatrixTest)))
    testError = RMSE(testPred, y_test[regressInd])
    if regressInd == 0:
        performancePlotGaussianGLM(xTrainValid, yTrainValid, trainPred, x_test[regressInd], y_test[0], testPred)
    print("Regression Test Error:", testError)
    return

def gaussian_glm_classification():
    xTrainValid = np.append(x_train[3], x_valid[3], axis = 0)
    yTrainValid = np.append(y_train[3], y_valid[3], axis = 0)
    minL, minS = findParameters(3)
    predMatrixTrain = createGaussianK(xTrainValid, xTrainValid, minS)
    predMatrixTest = createGaussianK(x_test[3], xTrainValid, minS)
    cholesky = sp.linalg.cho_factor(predMatrixTrain + minL * np.identity(len(predMatrixTrain)))
    alpha = sp.linalg.cho_solve(cholesky, yTrainValid)
    trainPred = np.transpose(np.matmul(np.transpose(alpha), np.transpose(predMatrixTrain)))
    testPred = np.transpose(np.matmul(np.transpose(alpha), np.transpose(predMatrixTest)))
    testPred = np.clip(np.around(testPred), 0, 2.1) 
    testError = incorrectPercentage(testPred, y_test[3])
    print("Classification Test Error: {}%".format(testError))
    return

def findParameters(dataSetInd):
    possibleLambdas = [1e-3, 1e-2, 0.1, 1]
    possibleThetas = [0.05, 0.1, 0.5, 1.0, 2.0]
    errorList = []
    if dataSetInd == 3:
        print("----------------------------")
        print("Classification Dataset:")
        print("Iris:")
        y_train[0] = np.expand_dims(intClassification(y_train[0]), axis = 1)
        y_valid[0] = np.expand_dims(intClassification(y_valid[0]), axis = 1)
        y_test[0] = np.expand_dims(intClassification(y_test[0]), axis = 1)
    else:
        print("----------------------------")
        print("Regression Dataset:")
        if dataSetInd == 0:
            print("Mauna_loa:")
        else:
            print("Rosenbrock:")
    for possibleL in possibleLambdas:
        minError = inf
        minPossibleL = None
        minPossibleS = None
        for possibleTheta in possibleThetas:
            predMatrixTrain = createGaussianK(x_train[dataSetInd], x_train[dataSetInd], possibleTheta)
            predMatrixValid = createGaussianK(x_valid[dataSetInd], x_train[dataSetInd], possibleTheta)
            cholesky = sp.linalg.cho_factor(predMatrixTrain + possibleL * np.identity(len(predMatrixTrain)))
            alpha = sp.linalg.cho_solve(cholesky, y_train[dataSetInd])

            yValidPred = np.transpose(np.matmul(np.transpose(alpha), np.transpose(predMatrixValid)))
            if dataSetInd == 3:
                yValidPred = np.clip(np.around(yValidPred), 0, 2.1)
            if dataSetInd == 3:
                error = incorrectPercentage(yValidPred, y_valid[dataSetInd])
            else:
                error = RMSE(yValidPred, y_valid[dataSetInd])
            if error < minError:
                minError = error
                minPossibleS = possibleTheta
                minPossibleL = possibleL
        errorList.append([minError, minPossibleL, minPossibleS])
    errorList = sorted(errorList, key=itemgetter(0))
    minL, minTheta = errorList[0][1], errorList[0][2]
    print("Minimum error:", errorList[0][0], "Lambda:", minL, "Theta:", minTheta)
    return minL, minTheta


if __name__ == "__main__":
   gaussian_glm_regression(1)
   gaussian_glm_regression(0)
   gaussian_glm_classification()