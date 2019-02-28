from data_utils import load_dataset
from svd_glm import sinFt, polyFt, expFt, RMSE, getPhi
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
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



def kernelized_cholesky_GLM():
    x_train_valid = np.expand_dims(np.append(x_train[0], x_valid[0]), axis=1)
    y_train_valid = np.expand_dims(np.append(y_train[0], y_valid[0]), axis=1)
    #Training:
    L = 10 # Min lambda found in svd_glm()
    phi = getPhi(x_train_valid)
    k = np.matmul(phi, np.transpose(phi))
    cholesky = sp.linalg.cho_factor(k + L*np.identity(len(k)))
    a = sp.linalg.cho_solve(cholesky, y_train_valid)
    y_train_valid_predictions = np.matmul(np.transpose(k), a)

    # Testing:
    testPhi = getPhi(x_test[0])
    k = np.matmul(phi, np.transpose(testPhi))
    y_test_predictions = np.matmul(np.transpose(k), a)

    # Error:
    error = RMSE(y_test_predictions, y_test[0])
    print("Error =", error)

    #Plotting:
    fig, ax = plt.subplots()
    zRange = np.expand_dims(np.arange(-0.1, 0.1, 0.001), axis=1)
    pZ = getPhi(zRange)
    pZ1 = getPhi(zRange + 1)
    p0 = getPhi(np.array([[0]]))  
    p1 = getPhi(np.array([[1]]))
    
    
    k0 = np.transpose(np.matmul(p0,np.transpose(pZ)))
    k1 =  np.transpose(np.matmul(p1,np.transpose(pZ1)))
    
    ax.plot(zRange, k0, 'mo', markersize = 0.25, label = 'k(0, z)')
    ax.plot(zRange, k1, 'bo', markersize = 0.25, label= 'k(1, z + 1)')
    plt.xlabel('Kernel Translational Invariant')
    plt.ylabel('Kernel Values')
    plt.title("Kernel vs Translational Invariant")
    plt.legend()
    plt.show()
    fig.savefig('results/KernelvsTranslationalInvariant.png')
    return

if __name__ == "__main__":
    kernelized_cholesky_GLM()

