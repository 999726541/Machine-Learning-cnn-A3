from PIL import Image
import numpy as np
import os
import csv
from shutil import copyfile


def PIL2array(img):
    return np.array(img.getdata(),
                    np.uint8).reshape(1, img.size[1] * img.size[0] * 3)


def PIL2array1(img):
    return np.array(img.getdata(),
                    np.uint8).reshape(img.size[1], img.size[0], 3)


def csv2dict(filename):
    dictionary = {}
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            dictionary[int(row['Id'])] = int(row['Label'])
    return dictionary


def getTargetArray(filename, dictionary):
    array = np.zeros((1, 8))
    array[0, dictionary[int(filename.split(".")[0])] - 1] = 1
    return array


def main():
    # print(result.shape)
    # print(np.sum(c != b))
    # print(result)
    dictionary = csv2dict('train.csv')
    path = 'train/'
    imgList = os.listdir(path)

    # Loading training set (6000)
    filename = imgList[0]
    input_train = PIL2array(Image.open(path + filename))
    target_train = getTargetArray(filename, dictionary)
    i = 0
    while i < 4900:  # len(imgList):
        print('Loading Image No. %d' % i)
        copyfile(path + imgList[i],
                 'data/training/' + str(dictionary[int(imgList[i].split(".")[0])]) + '/' + imgList[i])
        # input_train = np.concatenate((input_train, PIL2array(Image.open(path + imgList[i]))), axis = 0)
        # target_train = np.concatenate((target_train, getTargetArray(imgList[i], dictionary)), axis = 0)
        i += 1

    # Loading validation set
    filename = imgList[i]
    input_valid = PIL2array(Image.open(path + filename))
    target_valid = getTargetArray(filename, dictionary)
    # i += 1
    while i < len(imgList):
        print('Loading Image No. %d' % i)
        # input_valid = np.concatenate((input_valid, PIL2array(Image.open(path + imgList[i]))), axis = 0)
        # target_valid = np.concatenate((target_valid, getTargetArray(imgList[i], dictionary)), axis = 0)
        copyfile(path + imgList[i],
                 'data/validation/' + str(dictionary[int(imgList[i].split(".")[0])]) + '/' + imgList[i])
        i += 1

    print(input_train.shape)
    print(target_train.shape)
    print(input_valid.shape)
    print(target_valid.shape)
    print(np.sum(target_train, axis=0))
    print(np.sum(target_valid, axis=0))

    np.savez('data_1.npz', input_train=input_train, target_train=target_train, input_valid=input_valid,
             target_valid=target_valid)


if __name__ == '__main__':
    main()