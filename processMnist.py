# encoding: utf-8
"""
@author: monitor1379
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: Apache Licence
@file: mnist_decoder.py
@time: 2016/8/16 20:03

对MNIST手写数字数据文件转换为bmp图片文件格式。
数据集下载地址为http://yann.lecun.com/exdb/mnist。
相关格式转换见官网以及代码注释。

========================
关于IDX文件格式的解析规则：
========================
THE IDX FILE FORMAT

the IDX file format is a simple format for vectors and multidimensional matrices of various numerical types.
The basic format is

magic number
size in dimension 0
size in dimension 1
size in dimension 2
.....
size in dimension N
data

The magic number is an integer (MSB first). The first 2 bytes are always 0.

The third byte codes the type of the data:
0x08: unsigned byte
0x09: signed byte
0x0B: short (2 bytes)
0x0C: int (4 bytes)
0x0D: float (4 bytes)
0x0E: double (8 bytes)

The 4-th byte codes the number of dimensions of the vector/matrix: 1 for vectors, 2 for matrices....

The sizes in each dimension are 4-byte integers (MSB first, high endian, like in most non-Intel processors).

The data is stored like in a C array, i.e. the index in the last dimension changes the fastest.
"""

import numpy as np
import struct
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
# 训练集文件

train_images_idx3_ubyte_file = 'train-images.idx3-ubyte'
# 训练集标签文件
train_labels_idx1_ubyte_file = 'train-labels.idx1-ubyte'

# 测试集文件
test_images_idx3_ubyte_file = 't10k-images.idx3-ubyte'
# 测试集标签文件
test_labels_idx1_ubyte_file = 't10k-labels.idx1-ubyte'


def decode_idx3_ubyte(idx3_ubyte_file):
    """
    解析idx3文件的通用函数
    :param idx3_ubyte_file: idx3文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx3_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数、图片数量、每张图片高、每张图片宽
    offset = 0
    fmt_header = '>iiii'
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    print ('魔数:%d, 图片数量: %d张, 图片大小: %d*%d' % (magic_number, num_images, num_rows, num_cols))

    # 解析数据集
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(image_size) + 'B'
    images = np.empty((num_images, num_rows, num_cols))
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print ('已解析 %d' % (i + 1) + '张')
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        offset += struct.calcsize(fmt_image)
    return images


def decode_idx1_ubyte(idx1_ubyte_file):
    """
    解析idx1文件的通用函数
    :param idx1_ubyte_file: idx1文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx1_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数和标签数
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    print ('魔数:%d, 图片数量: %d张' % (magic_number, num_images))

    # 解析数据集
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print ('已解析 %d' % (i + 1) + '张')
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels


def load_train_images(idx_ubyte_file=train_images_idx3_ubyte_file):
    """
    TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  60000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
    Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

    :param idx_ubyte_file: idx文件路径
    :return: n*row*col维np.array对象，n为图片数量
    """
    return decode_idx3_ubyte(idx_ubyte_file)


def load_train_labels(idx_ubyte_file=train_labels_idx1_ubyte_file):
    """
    TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  60000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label
    The labels values are 0 to 9.

    :param idx_ubyte_file: idx文件路径
    :return: n*1维np.array对象，n为图片数量
    """
    return decode_idx1_ubyte(idx_ubyte_file)


def load_test_images(idx_ubyte_file=test_images_idx3_ubyte_file):
    """
    TEST SET IMAGE FILE (t10k-images-idx3-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  10000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
    Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

    :param idx_ubyte_file: idx文件路径
    :return: n*row*col维np.array对象，n为图片数量
    """
    return decode_idx3_ubyte(idx_ubyte_file)


def load_test_labels(idx_ubyte_file=test_labels_idx1_ubyte_file):
    """
    TEST SET LABEL FILE (t10k-labels-idx1-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  10000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label
    The labels values are 0 to 9.

    :param idx_ubyte_file: idx文件路径
    :return: n*1维np.array对象，n为图片数量
    """
    return decode_idx1_ubyte(idx_ubyte_file)

def MatrixProcess(n,pre):
    m=32/n
    m=int(m)
    post=np.zeros((n,n))
#将矩阵中的值变成0或1(0则不变，255则为1)
    for i in range(len(pre[0])):
        for j in range(len(pre[0])):
            if pre[i, j] == 0:
                pre[i, j] = 0
            else:
                pre[i, j] = 1
    for i in range(n):
        for j in range(n):
            # print (pre[j * m:(j + 1) * m, i * m:(i + 1) * m])
            # print (sum(pre[j * m:(j + 1) * m, i * m:(i + 1) * m]))
            if np.sum(pre[i*m:(i+1)*m,j*m:(j+1)*m])<=2:
                post[i,j]=0
            else:
                post[i,j]=1
    return post

def cut(n,data):
    # 图片切割
    flag = 0
    minx = n
    miny = n
    maxx = 0
    maxy = 0
    tempminx = 0
    tempminy = 0
    tempmaxx = n
    tempmaxy = n
    for i in range(n):
        for j in range(n):
            if data[i, j] != 0:
                flag += 1
                if flag == 1:
                    tempminx = i
                    tempminy = j
                tempmaxx = i
                tempmaxy = j
        if flag != 0:
            flag = 0
            if tempminx < minx:
                minx = tempminx
            if tempminy < miny:
                miny = tempminy
            if tempmaxx > maxx:
                maxx = tempmaxx
            if tempmaxy > maxy:
                maxy = tempmaxy
    # if maxy - miny < 50:
    #     maxy = maxy + 25
    #     miny = miny - 25
    # if maxx - minx < 50:
    #     maxx = maxx + 25
    #     minx = minx - 25

    scalim = Image.fromarray(data[minx:maxx, miny:maxy])
    scalim = scalim.resize((32, 32), Image.ANTIALIAS)
    #scalim.show()
    scalim = scalim.convert("L")
    scalim = scalim.convert("1")

    # clearNoise(im, 50, 2, 1)#除噪
    data_s = scalim.getdata()
    data_s = np.matrix(data_s)
    # 变换成512*512
    data_s = np.reshape(data_s, (32, 32))
    return data_s


def run():
    train_images = load_train_images("MNIST/t10k-images.idx3-ubyte")
    train_labels = load_train_labels("MNIST/t10k-labels.idx1-ubyte")
    # test_images = load_test_images()
    # test_labels = load_test_labels()
    #np.savetxt("mntrain.txt",train_images,fmt='%s')
    # 查看前十个数据及其标签以读取是否正确

    labelnum = np.zeros((1, 10))
    labelnum[0, 0] = 0
    labelnum[0, 1] = 0
    labelnum[0, 2] = 0
    labelnum[0, 3] = 0
    labelnum[0, 4] = 0
    labelnum[0, 5] = 22
    labelnum[0, 6] = 17
    labelnum[0, 7] = 15
    labelnum[0, 8] = 18
    labelnum[0, 9] = 17
    # labelnum[0, 0] = 7
    # labelnum[0, 1] = 8
    # labelnum[0, 2] = 7
    # labelnum[0, 3] = 6
    # labelnum[0, 4] = 6
    # labelnum[0, 5] = 10
    # labelnum[0, 6] = 6
    # labelnum[0, 7] = 7
    # labelnum[0, 8] = 11
    # labelnum[0, 9] = 10
    for i in range(600):
        print (train_labels[i])
        test_mat=cut(28, train_images[i])
        test=MatrixProcess(16,test_mat)
        print(test)
        np.savetxt("data/%d_%d.txt"%(int(train_labels[i]),labelnum[0,int(train_labels[i])]),test,'%d',newline='',delimiter='')
        labelnum[0, int(train_labels[i])]+=1
    print ('done')

if __name__ == '__main__':
    run()