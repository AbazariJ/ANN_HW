from numpy import *
import matplotlib.pyplot as plt
# import os
# import time


def hebb(s, t):
    '''Hebb'''
    N = len(s[0, :])
    M = len(t[0, :])
    P = len(s[:, 0])
    w = zeros([N, M], dtype=float)
    for p in range(P):
        w += dot(s[p, :].T.reshape(N, 1), t[p, :].reshape(1, M))
        # print('p={}\nw=\n{}\n'.format(p, w))

    return w


def perceptron(s, t, alpha=1.0, theta=0.0):
    ''' Perceptron '''
    N = len(s[0, :])
    M = len(t[0, :])
    P = len(s[:, 0])
    w = zeros([N, M], dtype=float)

    epoch = 0
    changes = 0
    for m in range(M):
        flag = True
        while flag and (epoch < 1000):
            flag = False
            for p in range(P):
                y_in = dot(s[p, :], w[:,m])
                if y_in > theta:
                    y = 1
                elif y_in < -theta:
                    y = -1
                else:
                    y = 0
                if y != t[p,m]:
                    w[:,m] += alpha * t[p,m] * s[p, :]

                    flag = True
                    changes+=1
            epoch += 1
        print('ipc = {}, chng = {}'.format(epoch,changes))
    print('w=\n{}'.format(w))
    return w


def delta(s, t, alpha=0.03):
    '''delta'''
    N = len(s[0, :])
    M = len(t[0, :])
    P = len(s[:, 0])
    w = zeros([N, M], dtype=float)
    # w = random.randn(N, M)

    epoch = 0
    alpha = 1/P
    tol = 1e-1*alpha
    error = 2 * tol

    for m in range(M):
        while (error > tol) and (epoch < 100):
            error = 0
            for p in range(P):
                y_in = dot(s[p, :], w[:, m])
                dw = alpha * (t[p, m] - y_in) * s[p, :]
                w[:, m] += dw
                if error < sqrt(dot(dw, dw)):
                    error = sqrt(dot(dw, dw))
            epoch += 1
            print('epoch={},error={}'.format(epoch, error))
    print('w=\n{}'.format(w))
    return w


def deltaEx(s, t, alpha=1, tol=1e-1):

    '''delta'''
    N = len(s[0, :])
    M = len(t[0, :])
    P = len(s[:, 0])
    # w = zeros([N, M], dtype=float)
    w = random.randn(N, M)

    epoch = 0

    error = 2 * tol
    gamma = 1
    for m in range(M):
        while (error > tol) and (epoch < 100):
        # while epoch < 100:
            error = 0
            for p in range(P):
                y_in = dot(s[p, :], w[:, m])
                dw = alpha * gamma * (t[p, m] - tanh(gamma * y_in)) * (1 - tanh(gamma * y_in)**2) * s[p, :]
                w[:, m] += dw
                if error < sqrt(dot(dw, dw)):
                    error = sqrt(dot(dw, dw))
            epoch += 1
            print('epoch={},error={}'.format(epoch, error))
    print('w=\n{}'.format(w))
    return w


def f(a, theta=0):
    return sign(a)

    # b = a.copy()
    # for i in range(len(b)):
    #     if a[i] > theta:
    #         b[i] = 1
    #     elif a[i] < -theta:
    #         b[i] = -1
    #     else:
    #         b[i] = 0
    # return b
