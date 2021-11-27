import numpy as np
import h5py
import matplotlib.pyplot as plt

np.random.seed(42)

def initialize_parameters(n_x, n_h, n_y):
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    parameters = {"W1": W1, 
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    return parameters

def initialize_parameters_deep(layer_dims):
    parameters = {}
    L = len(layer.dims)

    for i in range(1, L):
        parameters["W" + str(i)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters["b" + str(i)] = np.zeros((layer_dims[l], 1))
    return parameters

def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    return Z, cache

def sigmoid(z):
    a = 1 / (1 + np.exp(-z))
    return a, z

def relu(z):
    a = np.max(0, z)
    return a, z

def linear_activation_forward(A_prev, W, b, activation):
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    cache = (linear_cache, activation_cache)
    return A, cache

def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2
    
    for l in range(1, L):
        A_prev = A

        A, cache = linear_activation_forward(A_prev, parameters["W" \
                + str(l)], parameters["b"+str(l)], activation="relu")
        caches.append(cache)

    AL, cache = linear_activation_forward(A, parameters["W" + \
            str(l)], parameters["b"+str(l)], activation="sigmoid")
    caches.append(cache)
    return AL, caches

def compute_cost(AL, Y):
    m = Y.shape[1]

    cost = (-1/m) * np.sum(np.multipy(Y, np.log(AL)), np.multiply((1-Y), np.log(1-AL)))
    cost = np.squeeze(cost)
    return cost


