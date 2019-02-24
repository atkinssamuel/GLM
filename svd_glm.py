from data_utils import load_dataset
import matplotlib.pyplot as plt
import numpy as np
import math

x_train, x_valid, x_test, y_train, y_valid, y_test = ([[],[],[],[],[]], [[],[],[],[],[]],
                                                     [[],[],[],[],[]], [[],[],[],[],[]],
                                                     [[],[],[],[],[]], [[],[],[],[],[]])
x_train[0], x_valid[0], x_test[0], y_train[0], y_valid[0], y_test[0] = list(load_dataset('mauna_loa'))
x_train[1], x_valid[1], x_test[1], y_train[1], y_valid[1], y_test[1] = list(load_dataset('rosenbrock', n_train=5000, d=2))
x_train[2], x_valid[2], x_test[2], y_train[2], y_valid[2], y_test[2] = list(load_dataset('pumadyn32nm'))
x_train[3], x_valid[3], x_test[3], y_train[3], y_valid[3], y_test[3] = list(load_dataset('iris'))
x_train[4], x_valid[4], x_test[4], y_train[4], y_valid[4], y_test[4] = list(load_dataset('mnist_small'))
inf = 10000000


def RMSE(measurements, actuals):
    n = len(measurements)
    sum = 0
    for i in range(n):
        sum += (measurements[i] - actuals[i])**2

    return math.sqrt(sum/n)

def expFt(x):
    return np.exp(x)

def sinFt(x, a): 
    x = np.array(x)
    return np.sin(a*x)

def polyFt(x, p): 
    x = np.array(x) 
    return np.power(x, p)

def getPhi(x):
    x = np.array(x)
    phi = np.ones((len(x), 1))
    for p in range(6):
        phi = np.append(phi, polyFt(x, p), axis = 1)
    phi = np.append(phi, sinFt(x, 110), axis = 1)
    return phi

def minLambda():
    possibleLambdaValues = [1e-8, 1e-7, 1e-6, 1e-5, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100, 1000]
    phiInit = getPhi(x_train[0])
    minErr = inf
    minL = None
    for possibleL in possibleLambdaValues:
        phi = phiInit
        weights = computeWeights(phi, possibleL, y_train[0])
        
        phi = getPhi(x_valid[0])   
        predictions = np.matmul(phi, weights)

        error = RMSE(predictions, y_valid[0])
        print(error)
        if error < minErr:
            minErr = error
            minL = possibleL
    return minL

def computeWeights(phi, L, y):
    U, S, VT = np.linalg.svd(phi, full_matrices=False)
    V = np.transpose(VT) 
    S = np.diag(S)
    STS = np.matmul(np.transpose(S), S)
    
    firstHalf = np.matmul(V, np.linalg.inv(STS + L*np.identity(len(S))))
    weights = np.matmul(np.matmul(np.matmul(firstHalf, np.transpose(S)), np.transpose(U)), y)
    return weights

def performancePlot(x_train_valid, y_train_valid, train_valid_predictions, xTest, yTest, predictions):
    fig, ax = plt.subplots()
    ax.plot(x_train_valid, y_train_valid, 'go', markersize = 1, label='Training and Validation Data')
    ax.plot(x_train_valid, train_valid_predictions, 'bo', markersize = 0.5, label='Training and Validation Prediction')
    ax.plot(xTest, yTest, 'co', markersize = 1, label='Test Data')
    ax.plot(xTest, predictions, 'mo', markersize = 0.5, label='Test Prediction')
    ax.legend()
    plt.xlabel('x Values')
    plt.ylabel('y Values')
    plt.title('Training and Testing Actuals and Predictions')
    fig.savefig('TrainingTestingPredictions.png')
    plt.show()
    return

def svd_glm():
    L = minLambda()
    print("Min Lambda =", L)
    x_train_valid = np.expand_dims(np.append(x_train[0], x_valid[0]), axis=1)
    y_train_valid = np.expand_dims(np.append(y_train[0], y_valid[0]), axis=1)
    phi = getPhi(x_train_valid)
    weights = computeWeights(phi, L, y_train_valid)
    train_valid_predictions = np.matmul(phi, weights)
    phi = getPhi(x_test[0])
    test_predictions = np.matmul(phi, weights)
    error = RMSE(test_predictions, y_test[0])
    performancePlot(x_train_valid, y_train_valid, train_valid_predictions, x_test[0], y_test[0], test_predictions)
    print("Error =", error)
    return error

if __name__ == "__main__":
    svd_glm()