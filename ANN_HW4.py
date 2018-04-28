from MyPakage import *
import idx2numpy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def problem_1():
    pass

def problem_2():
    pass

def problem_3():
    # s = idx2numpy.convert_from_file('./data/train-images.idx3-ubyte')
    # t = idx2numpy.convert_from_file('./data/train-labels.idx1-ubyte')
    s = idx2numpy.convert_from_file('./data/t10k-images.idx3-ubyte')
    t = idx2numpy.convert_from_file('./data/t10k-labels.idx1-ubyte')
    # plt.hist(t)
    # plt.show()
    # print(s[1][:][:])
    # row = 5
    # col = 5
    # oFig1 = plt.figure(1)
    # for i in range(row * col):
    #     I = i
    #     a = s[I][:][:]
    #     # print(t[I])
    #     oFig1.add_subplot(row, col, i+1)
    #     plt.axis('off')
    #     plt.imshow(a)
    # plt.show()

    # I = 0
    # i = 100
    # row = 2
    # col = 5
    # oFig2 = plt.figure(2)
    # while I < 10:
    #     if t[i] == I:
    #         a = s[i][:][:]
    #         # print(t[i])
    #         oFig2.add_subplot(row, col, I + 1)
    #         plt.axis('off')
    #         plt.imshow(a)
    #         I += 1
    #     i += 1
    # plt.show()
    # plt.tight_layout()


    row = 5
    col = 5
    for I in range(10):
        oFig = plt.figure(I)
        num = 0
        i = 0
        while num < row*col:
            if t[i] == I:
                a = s[i][:][:]
                plt.imshow(a)
                num += 1
            i += 1
        plt.show()


def main():


    problem_3()


if __name__ == "__main__":
    main()