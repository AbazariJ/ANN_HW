from MyPakage import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def hamming_distance(s1: array, s2: array):
    size = len(s1)
    return (size-dot(s1, s2))/(2*size)

def problem_1():
    s = array([[1, 1, 1, 1],
               [1, 1,-1,-1]])

    w1 = hebb(s, s)
    w2 = w1.copy()
    fill_diagonal(w2, 0)
    print('w1=\n {}\n w2=\n {}'.format(w1, w2))

    v1 = array([[1, 1, 1, 1]])
    v2 = array([[1, 1,-1,-1]])
    v3 = array([[1, 1, 1, 0]])

    print('non-zero diagonal weight matrix:\ny(1)=\n{}\ny(2)=\n{}\ny(3)=\n{}\n'.format(f(dot(v1, w1)), f(dot(v2, w1)), f(dot(v3, w1))))
    print('set zero diagonal weight matrix:\ny(1)=\n{}\ny(2)=\n{}\ny(3)=\n{}\n'.format(f(dot(v1, w2)), f(dot(v2, w2)), f(dot(v3, w2))))


def problem_2():
    s = array([
        [1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1,
         1, 1,-1,-1,-1,-1, 1, 1,
         1, 1,-1,-1,-1,-1, 1, 1,
         1, 1,-1,-1,-1,-1, 1, 1,
         1, 1,-1,-1,-1,-1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1],

        [-1,- 1, 1, 1, 1,-1,-1,-1,
         -1,-1, 1, 1, 1,-1,-1,-1,
         -1,-1,-1, 1, 1,-1,-1,-1,
         -1,-1,-1, 1, 1,-1,-1,-1,
         -1,-1,-1, 1, 1,-1,-1,-1,
         -1,-1,-1, 1, 1,-1,-1,-1,
         -1,-1,-1, 1, 1,-1,-1,-1,
         -1,-1,-1, 1, 1,-1,-1,-1]
    ])

    w = hebb(s, s)
    fill_diagonal(w, 0)

    plt.figure(1)
    for i in range(2):
        index = random.permutation(64)
        index = index[0:int(0.3 * 64)]

        s_p = s[i][:].copy()
        s_p[index] *= -1

        output = f(dot(s_p, w))

        plt.subplot(221 + 2 * i)
        # plt.spy((s_p + 1).reshape(8, 8))
        plt.imshow((s_p + 1).reshape(8, 8))

        plt.subplot(222 + 2 * i)
        # plt.spy((output + 1).reshape(8, 8))
        plt.imshow((output + 1).reshape(8, 8))

    plt.show()

    print(hamming_distance(s[0][:], s[1][:]))


    # print('noisy input is')
    # for j in range(8):
    #     for i in range(8):
    #         print(int((s_p[i+j*8]+1)/2),end='')
    #     print('')
    # print('')
    #
    #
    # print('patern detected')
    # for j in range(8):
    #     for i in range(8):
    #         print(int((output[i+j*8]+1)/2),end='')
    #     print('')
    # print('')

    # print(index)
    #
    # plt.spy(w)
    # plt.show()


def problem_3():
    s = array([
        # '''A'''
        [-1, 1,-1,
          1,-1, 1,
          1, 1, 1,
          1,-1, 1,
          1,-1, 1],
        # '''B'''
        [ 1, 1,-1,
          1,-1, 1,
          1, 1,-1,
          1,-1, 1,
          1, 1,-1],
        # '''C'''
        [-1, 1, 1,
          1,-1,-1,
          1,-1,-1,
          1,-1,-1,
         -1, 1, 1],
        # '''D'''
        [ 1, 1,-1,
          1,-1, 1,
          1,-1, 1,
          1,-1, 1,
          1, 1,-1],
        # '''E'''
        [ 1, 1, 1,
          1,-1,-1,
          1, 1,-1,
          1,-1,-1,
          1, 1, 1],
        # '''F'''
        [ 1, 1, 1,
          1,-1,-1,
          1, 1,-1,
          1,-1,-1,
          1,-1,-1],
        # '''G'''
        [-1, 1, 1,
          1,-1,-1,
          1,-1, 1,
          1,-1, 1,
         -1, 1, 1],
        # '''H'''
        [ 1,-1, 1,
          1,-1, 1,
          1, 1, 1,
          1,-1, 1,
          1,-1, 1]
    ])
    t = array([
        [-1,-1,-1],#A
        [-1,-1, 1],#B
        [-1, 1,-1],#C
        [-1, 1, 1],#D
        [ 1,-1,-1],#E
        [ 1,-1, 1],#F
        [ 1, 1,-1],#G
        [ 1, 1, 1],#H
    ])
    '''input vector visualize'''
    for i in range(8):
        a = s[i][:]
        plt.subplot(241 + i)
        # plt.spy((a + 1).reshape(5, 3))
        plt.imshow((a).reshape(5, 3))
    plt.show()

    # s1 = s[[0,2]][:]
    # t1 = array([[-1,1],[1,1]])
    # w1 = hebb(s1,t1)
    # print(w1)

    w=hebb(s, t)
    # w = perceptron(s, t)
    # w = deltaEx(s, t, 0.01,1e-5)

    print(f(dot(s, w)) - t)
    print(f(dot(t, w.T)))
    oFig = plt.figure(1)
    for i in range(8):
        a = f(dot(t[i][:], w.T))
        # plt.subplot(241 + i)
        # plt.spy(((a + 1)).reshape(5, 3))
        oFig.add_subplot(2, 4, i + 1)
        plt.imshow((a).reshape(5, 3))
        plt.axis('off')
        plt.colorbar()
    plt.show()
    plt.tight_layout()

    '''hammming distance'''
    for i in range(8):
        for j in range(i+1,8):
            inp=(15-dot(s[i][:],s[j][:]))/30
            out=(3-dot(t[i][:],t[j][:]))/6
            print('{},{}'.format(i, j))
            print('input ={}'.format(inp))
            print('outpt ={}'.format(out))
            print(100*(inp/out if out>inp else out/inp),'\n___________________________')




def main():

    # problem_1()
    problem_2()
    # problem_3()

if __name__ == "__main__":
    main()