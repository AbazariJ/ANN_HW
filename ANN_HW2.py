from numpy import *
import matplotlib.pyplot as plt
import os
import time


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


def my_plot(s, t, w=None, lim=5):
    # print(w)
    plt.scatter(s[:, 0], s[:, 1], c=t.reshape(len(s[:, 0])))

    if w is not None:
        for i in range(len(w[0, :])):
            Yp = 0.2 * sqrt(w[0, i] ** 2 + w[1, i] ** 2)
            plt.plot([-lim, lim], [-(-lim * w[0,i] + w[2,i]) / w[1,i], -(lim * w[0,i] + w[2,i]) / w[1,i]],'b')
            plt.plot([-lim, lim], [-(-lim * w[0,i] + w[2,i]+Yp) / w[1,i], -(lim * w[0,i] + w[2,i]+Yp) / w[1,i]],':r')

    # plt.plot([-lim, lim], [-(-lim * w[0,0] + w[2,0]) / w[1,0], -(lim * w[0,0] + w[2,0]) / w[1,0]],'b')
    # plt.plot([-lim, lim], [-(-lim * w[0,1] + w[2,1]) / w[1,1], -(lim * w[0,1] + w[2,1]) / w[1,1]],'r')
    # if lim is not None:
    plt.xlim(-0.75 * lim, 0.75 * lim)
    plt.ylim(-0.75 * lim, 0.75 * lim)
    plt.xlabel('X1')
    plt.ylabel('X2')
    # plt.axis('equal')
    plt.show()


def MRI(s, t, alpha=0.5, theta=0):
    N = len(s[0, :])
    M = len(t[0, :])
    P = len(s[:, 0])
    w = random.randn(N, 2)
    # w = array([[0.05, 0.1], [0.2, 0.2], [0.3, 0.15]])
    # w = array([[-0.73, 1.27], [1.53, -1.33], [-0.99, -1.09]])
    v = array([[0.5], [0.5], [0.5]])
    # z = zeros([2, 1], dtype=float)
    epoch = 0
    changes = 0

    # plot(s, t, w)
    for m in range(M):
        flag = True
        while flag and (epoch < 100):
            flag = False
            for p in range(P):
                z_in = dot(s[p, :], w)
                z = f(z_in,0.1)
                y_in = dot(append(z, 1), v)
                # y = f(y_in, theta)
                y = f(y_in,0.1)
                if y != t[p, m]:
                    if t[p, m] > 0:
                        #for i in range(2):
                        i = argmin(abs(z_in))
                        w[:, i] += alpha * (t[p, m] - z_in[i]) * s[p, :]
                        # print('s[p,:]={},t[p,m]={},z_in={},i={}'.format(s[p,:],t[p,m],z_in,i))
                    else:
                        for i in nonzero(z > 0.0)[0]:
                            w[:, i] += alpha * (t[p, m] - z_in[i]) * s[p, :]
                    # print(s[p,0],s[p,1])
                    # print(w)
                    # plot(s, t, w)
                    flag = True
                    changes += 1

            epoch += 1
        print('ipc = {}, chng = {}'.format(epoch, changes))
    return w


def MRI_3line(s, t, alpha=0.5, theta=0):
    N = len(s[0, :])
    M = len(t[0, :])
    P = len(s[:, 0])
    w = random.randn(N, 3)
    # w = array([[0.05, 0.1, 1], [0.2, 0.2, 1], [0.3, 0.15, 1]])
    v = array([[1/3], [1/3], [1/3], [2/3]])
    # z = zeros([3, 1], dtype=float)
    epoch = 0
    changes = 0

    # plot(s, t, w)
    for m in range(M):
        flag = True
        while flag and (epoch < 100):
            flag = False
            for p in range(P):
                z_in = dot(s[p, :], w)
                z = f(z_in, 0.1)
                y_in = dot(append(z, 1), v)
                # y = f(y_in, theta)
                y = f(y_in, 0.1)
                if y != t[p, m]:
                    if t[p, m] > 0:
                        i = argmin(abs(z_in))
                        w[:, i] += alpha * (t[p, m] - z_in[i]) * s[p, :]
                        # print('s[p,:]={},t[p,m]={},z_in={},i={}'.format(s[p,:],t[p,m],z_in,i))
                    else:
                        for i in nonzero(z > 0.0)[0]:
                            w[:, i] += alpha * (t[p, m] - z_in[i]) * s[p, :]
                            # print('s[p,:]={},t[p,m]={},z_in={},i={}'.format(s[p, :], t[p, m], z_in, i))
                    # print(s[p,0],s[p,1])
                    # print(w)
                    # plot(s, t, w)
                    flag = True
                    changes += 1

            epoch += 1
        print('ipc = {}, chng = {}'.format(epoch, changes))
    return w


def MRI_MultiLine(s, t, Num=3 , alpha=0.5, theta=0):
    N = len(s[0, :])
    M = len(t[0, :])
    P = len(s[:, 0])
    w = random.randn(N, Num)
    v = append(ones([Num])/Num, (Num-1)/Num).reshape(Num+1, 1)
    # v = array([[1/3], [1/3], [1/3], [2/3]])
    # z = zeros([3, 1], dtype=float)
    epoch = 0
    changes = 0

    # plot(s, t, w)
    for m in range(M):
        flag = True
        while flag and (epoch < 100):
            flag = False
            for p in range(P):
                z_in = dot(s[p, :], w)
                z = f(z_in, 0.1)
                y_in = dot(append(z, 1), v)
                y = f(y_in, 0.1)
                if y != t[p, m]:
                    if t[p, m] > 0:
                        i = argmin(abs(z_in))
                        w[:, i] += alpha * (t[p, m] - z_in[i]) * s[p, :]
                    else:
                        for i in nonzero(z > 0.0)[0]:
                            w[:, i] += alpha * (t[p, m] - z_in[i]) * s[p, :]
                    flag = True
                    changes += 1

            epoch += 1
        print('ipc = {}, chng = {}'.format(epoch, changes))
    return w


def madaline(s, t, alpha=1):
    '''madaline'''
    N = len(s[0, :])
    M = len(t[0, :])
    P = len(s[:, 0])
    w = random.randn(N, 2)
    v = array([[0.5], [0.5], [0.5]])
    epoch = 0
    tol = 1e-1
    error = 2 * tol
    gamma = 1
    for m in range(M):
        while (error > tol) and (epoch < 15):
            # while epoch < 100:
            error = 0
            for p in range(P):
                for k in range(2):
                    z_in = dot(s[p, :], w)
                    z = append(tanh(z_in), 1)
                    y_in = dot(z, v)
                    dw = alpha * (gamma ** 2) * (t[p, m] - tanh(gamma * y_in)) *\
                         (1 - tanh(gamma * y_in) ** 2) * (1 - z[k]) * s[p, :]
                    w[:, k] += dw
                    if error < sqrt(dot(dw, dw)):
                        error = sqrt(dot(dw, dw))
            epoch += 1
            print('epoch={},error={}'.format(epoch, error))

            lim = 10
            plt.scatter(s[:, 0], s[:, 1], c=t.reshape(4))
            # plt.plot([-lim, lim], [-(-lim * w[0] + w[2] +teta) / w[1], -(lim * w[0] + w[2]+teta) / w[1]])
            plt.plot([-lim, lim], [-(-lim * w[0, 0] + w[2, 0]) / w[1, 0], -(lim * w[0, 0] + w[2, 0]) / w[1, 0]])
            plt.plot([-lim, lim], [-(-lim * w[0, 1] + w[2, 1]) / w[1, 1], -(lim * w[0, 1] + w[2, 1]) / w[1, 1]])
            plt.xlim(-0.75 * lim, 0.75 * lim)
            plt.ylim(-0.75 * lim, 0.75 * lim)
            plt.xlabel('X1')
            plt.ylabel('X2')
            plt.show()

            # wait = input("PRESS ENTER TO CONTINUE.")
            # os.system("pause")
    print('w=\n{}'.format(w))
    return w


def HW2_1():
    '''HW2_1'''
    s = array([[-1, 1,-1, -1, 1,-1, -1, 1, 1,  1], #L
             [-1, 1,-1, -1, 1,-1, -1, 1,-1,  1], #I
             [ 1, 1, 1,  1,-1, 1,  1, 1, 1,  1], #O
             [ 1,-1, 1,  1,-1, 1,  1, 1, 1,  1]])#U
    t = array([[-1],
             [-1],
             [-1],
             [+1]])
    #
    # s = array([[1, 0, 1, 1, 0, 1, 1, 1, 1, 1],
    #            [0, 1, 0, 0, 1, 0, 0, 1, 0, 1],
    #            [1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
    #            [0, 1, 0, 0, 1, 0, 0, 1, 1, 1]])
    # t = array([[1],
    #            [-1],
    #            [-1],
    #            [-1]])
    w = perceptron(s, t)
    # w = deltaEx(s, t)

    # w = hebb(s, t)
    print('w=\n{}'.format( w))
    print('s*w=\n{}'.format(dot(s, w)))


def HW2_2():
    '''HW2_2'''
    s = array([[1, 1, 1, 1],
               [-1, -1, 1, 1],
               [1, 1, -1, 1]])
    t = array([[1, -1, -1],
               [-1, 1, -1],
               [-1, -1, 1]])

    # s = array([[1, 1, 1, 1],
    #            [0, 0, 1, 1],
    #            [1, 1, 0, 1]])
    # t = array([[1, -1, -1],
    #            [-1, 1, -1],
    #            [-1, -1, 1]])

    w = perceptron(s, t)
    # w = deltaEx(s, t, 0.5, 1e-2)

    # w = hebb(s, t)
    print('w=\n{}'.format( w))
    print('s*w=\n{}'.format(dot(s, w)))


def HW2_3():
    '''HW2_3'''
    Q1 = 1000
    Q2 = 10
    a, b = 2, 0.5
    x1 = +a + b * random.randn(Q1)
    y1 = +a + b * random.randn(Q1)
    x2 = -a + b * random.randn(Q2)
    y2 = -a + b * random.randn(Q2)
    s = array([append(x1, x2), append(y1, y2), ones(Q1 + Q2)]).T
    t = append(ones([Q1]), -ones([Q2])).reshape(Q1 + Q2, 1)

    teta = 0.2
    # w = perceptron(s, t, 1, teta)
    # w = delta(s, t)
    w = deltaEx(s, t)

    # w = hebb(s, t)
    # print('w=\n{}'.format(w))
    # print('s*w=\n{}'.format(dot(s, w)))

    # lim = 5
    # plt.scatter(s[:, 0], s[:, 1], c=t.reshape(Q1 + Q2))
    # # plt.plot([-lim, lim], [-(-lim * w[0] + w[2] +teta) / w[1], -(lim * w[0] + w[2]+teta) / w[1]])
    # plt.plot([-lim, lim], [-(-lim * w[0] + w[2]) / w[1], -(lim * w[0] + w[2]) / w[1]])
    # print( -(-lim * w[0] + w[2]) / w[1], -(lim * w[0] + w[2]) / w[1])
    # # plt.plot([-lim, lim], [-(-lim * w[0] + w[2] -teta) / w[1], -(lim * w[0] + w[2]-teta) / w[1]])
    # plt.xlim(-0.75 * lim, 0.75 * lim)
    # plt.ylim(-0.75 * lim, 0.75 * lim)
    # plt.xlabel('X1')
    # plt.ylabel('X2')
    # plt.show()
    my_plot(s, t, w)


def HW2_4():
    s = array([[0, 0, 1],
               [0, 1, 1],
               [-1, -1, 1],
               [+1, -1, 1]])
    t = array([[-1],
               [+1],
               [+1],
               [+1]])

    w = MRI_3line(s, t)
    # w = MRI(s, t)
    print('s*w*v=\n{}'.format((f(dot(s, w)))))
    my_plot(s, t, w, 3)
    print('w=',w)


def HW2_5a():
    """a"""
    s = array([[0, 0, 1],
               [1, 0, 1],
               [2, 0, 1],
               [0, 1, 1],
               [1, 1, 1],
               [2, 1, 1],
               [0, 2, 1],
               [1, 2, 1],
               [2, 2, 1]])
    t = array([[+1],
               [+1],
               [+1],
               [+1],
               [-1],
               [+1],
               [+1],
               [+1],
               [+1]])
    # s = array([[0, 0, 1],
    #            [1, 0, 1],
    #            [2, 0, 1],
    #            [3, 0, 1],
    #            [0, 1, 1],
    #            [1, 1, 1],
    #            [2, 1, 1],
    #            [3, 1, 1],
    #            [0, 2, 1],
    #            [1, 2, 1],
    #            [2, 2, 1],
    #            [3, 2, 1]])
    # t = array([[+1],
    #            [+1],
    #            [-1],
    #            [-1],
    #            [+1],
    #            [+1],
    #            [-1],
    #            [-1],
    #            [+1],
    #            [+1],
    #            [-1],
    #            [+1]])

    z = zeros([len(s[:, 0]), 5])
    z[:, 0] = s[:, 0]**2
    z[:, 1] = s[:, 1]**2
    z[:, 2] = s[:, 0]
    z[:, 3] = s[:, 1]
    z[:, 4] = s[:, 2]
    # print('z=', z)
    # w = deltaEx(z, t, 1, 0.001)
    w = perceptron(z, t)
    print('w=\n', w)
    print('z*w=\n{}'.format(dot(z, w)))


    # w/=(w[2, 0]**2 + w[3, 0]**2 -4)/(4*w[4, 0])
    a, b, c, d, e = w[0], w[1], w[2], w[3], w[4]
    # a, b, c, d, e =1,1,-2,-2,1
    alp = 0.25 * c * c / a + 0.25 * d * d / b - e
    A, B = sqrt(alp / a), sqrt(alp / b)
    X0, Y0 = -0.5 * c / a, -0.5 * d / b
    tet = arange(-0.1, 2 * pi, 0.1)
    X = X0 + A * cos(tet)
    Y = Y0 + B * sin(tet)
    plt.plot(X, Y)


    # x = arange(-2.0, 6.0, 0.001)
    # a, b, c = w[1, 0], w[3, 0], w[0, 0]*x**2 + w[2, 0]*x + w[4, 0]
    # y = append((-b+sqrt(b**2-4*a*c))/(2*a), (-b-sqrt(b**2-4*a*c))/(2*a))
    # x = append(x, x)
    # plt.plot(x, y)
    plt.xlim(-0.25, 2.25)
    plt.ylim(-0.25, 2.25)
    my_plot(s, t)
    plt.show()


def HW2_5b():
    """b"""
    s = array([[1, 1, 1],
               [1, -1, 1],
               [0, 0, 1],
               [-1, 1, 1],
               [-1, -1, 1]])
    t = array([[+1],
               [+1],
               [-1],
               [+1],
               [+1]])

    z = zeros([len(s[:, 0]), 3])
    z[:, 0] = s[:, 0]**2
    z[:, 1] = s[:, 1]**2
    z[:, 2] = s[:, 2]
    # print('z=', z)
    # w = deltaEx(z, t, 1, 0.1)
    w = perceptron(z, t)
    print('w=\n',w)
    print('z*w=\n{}'.format(dot(z, w)))

    a, b, c = w[0], w[1], w[2]
    A, B = sqrt(-c / a), sqrt(-c / b)
    tet = arange(-0.1, 2 * pi, 0.1)
    X = A * cos(tet)
    Y = B * sin(tet)
    plt.plot(X, Y)
    plt.xlim(-1.25, 1.25)
    plt.ylim(-1.25, 1.25)
    my_plot(s, t)
    plt.show()


def HW2_5c():
    """b"""
    s = array([[1, 1, 1],
               [1, -1, 1],
               [0, 0, 1],
               [-1, 1, 1],
               [-1, -1, 1],
               [1, 0, 1]])
    t = array([[+1],
               [+1],
               [-1],
               [+1],
               [+1],
               [-1]])

    z = zeros([len(s[:, 0]), 2])
    z[:, 0] = s[:, 1]**2
    z[:, 1] = s[:, 2]
    # print('z=', z)
    # w = deltaEx(z, t, 1, 0.1)
    w = perceptron(z, t)
    print('w=\n',w)
    print('z*w=\n{}'.format(dot(z, w)))

    plt.plot([-2, 2], [+sqrt(-w[1]/w[0]), +sqrt(-w[1]/w[0])])
    plt.plot([-2, 2], [-sqrt(-w[1]/w[0]), -sqrt(-w[1]/w[0])])
    plt.xlim(-1.25, 1.25)
    plt.ylim(-1.25, 1.25)
    my_plot(s, t)
    plt.show()


def HW2_5e():
    Q1 = 6
    Q2 = Q1
    Q3 = Q1
    Q4 = Q1
    Q5 = Q1
    a, b = 1, 0.1

    '''set1'''
    R = +a + b * random.randn(Q1)
    teta = random.rand(Q1)*2*pi
    x1 = R * cos(teta)
    y1 = R * sin(teta)

    x2 = + b * random.randn(Q2)
    y2 = + b * random.randn(Q2)

    s = array([append(x1, x2), append(y1, y2),ones(Q1 + Q2)]).T
    t = append(ones([Q1]), -ones([Q2])).reshape(Q1 + Q2, 1)

    '''set2'''
    # x1 = +a + b * random.randn(Q1)
    # y1 = +a + b * random.randn(Q1)
    #
    # x2 = -a + b * random.randn(Q2)
    # y2 = +a + b * random.randn(Q2)
    #
    # x3 = -a + b * random.randn(Q2)
    # y3 = -a + b * random.randn(Q2)
    #
    # x4 = +a + b * random.randn(Q2)
    # y4 = -a + b * random.randn(Q2)
    #
    # x5 = + b * random.randn(Q4)
    # y5 = + b * random.randn(Q4)
    #
    # s = array([append(append(append(append(x1, x2),x3),x4),x5), append(append(append(append(y1, y2),y3),y4),y5), ones(Q1 + Q2 +Q3 + Q4 + Q5)]).T
    # t = append(ones([Q1 + Q2+Q3 + Q4]), -ones([Q5])).reshape(Q1 + Q2 +Q3 + Q4 + Q5, 1)

    '''set3'''
    # R = +a *ones(Q1).T
    # teta = arange(0, 1, 1/Q1)*2*pi
    # x1 = R * cos(teta)
    # y1 = R * sin(teta)
    # x2 = 0.8*x1
    # y2 = 0.8*y1
    # s = array([append(x1, x2), append(y1, y2),ones(Q1 + Q2)]).T
    # t = append(ones([Q1]), -ones([Q2])).reshape(Q1 + Q2, 1)

    w = MRI_3line(s, t)
    my_plot(s, t, w, 8)

    w = MRI_MultiLine(s, t, 10)
    my_plot(s, t, w, 8)

    z = zeros([len(s[:, 0]), 5])
    z[:, 0] = s[:, 0] ** 2
    z[:, 1] = s[:, 1] ** 2
    z[:, 2] = s[:, 0]
    z[:, 3] = s[:, 1]
    z[:, 4] = s[:, 2]
    # print('z=', z)
    # w = deltaEx(z, t, 1, 0.001)
    w = perceptron(z, t)
    # print('w=\n', w)
    # print('z*w=\n{}'.format(dot(z, w)))

    # w/=(w[2, 0]**2 + w[3, 0]**2 -4)/(4*w[4, 0])
    a, b, c, d, e = w[0], w[1], w[2], w[3], w[4]
    # a, b, c, d, e =1,1,-2,-2,1
    alp = 0.25 * c * c / a + 0.25 * d * d / b - e
    A, B = sqrt(alp / a), sqrt(alp / b)
    X0, Y0 = -0.5 * c / a, -0.5 * d / b
    tet = arange(-0.1, 2 * pi, 0.1)
    X = X0 + A * cos(tet)
    Y = Y0 + B * sin(tet)
    plt.plot(X, Y)
    # plt.xlim(-0.25, 2.25)
    # plt.ylim(-0.25, 2.25)
    my_plot(s, t, None, 8)
    plt.show()

def tests():
    '''AND test'''
    # s = array([[1, 1,  1],
    #          [1, 0,  1],
    #          [0, 1,  1],
    #          [0, 0,  1]])
    # t = array([[+1],
    #            [-1],
    #            [-1],
    #            [-1]])
    #
    # s = array([[1, 1, 1],
    #            [1, -1, 1],
    #            [-1, 1, 1],
    #            [-1, -1, 1]])
    # t = array([[1],
    #            [-1],
    #            [-1],
    #            [-1]])

    '''XOR test'''
    s = array([[1, 1, 1],
               [1, -1, 1],
               [-1, 1, 1],
               [-1, -1, 1]])
    t = array([[-1],
               [+1],
               [+1],
               [-1]])

    w = MRI(s,t)
    # w = myMRI(s,t)

    my_plot(s, t, w, 10)


def main():

    # test()
    # HW2_1()
    # HW2_2()
    # HW2_3()
    # HW2_4()
    HW2_5e()
    # HW2_5b()
    # HW2_5c()
    # s = array([[-1, -1, 1]])
    # t = array([[-1]])
    # w=array([[1], [1], [-1]])
    # my_plot(s,t,w)
    # print(dot(s,w))


if __name__ == "__main__":
    main()
