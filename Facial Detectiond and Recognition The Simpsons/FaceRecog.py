from Parameters import *
import numpy as np
from sklearn.svm import SVC, LinearSVC
import matplotlib.pyplot as plt
import glob
import cv2 as cv
import timeit
import PIL
import copy
from sklearn import svm
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split


class FaceRecog:
    def __init__(self, params:Parameters):
        self.params = params
        self.svm_classifier = svm.SVC(C=3.5) # definirea modelului

    def train(self):
        train_images = []
        train_labels = []

        names  = ["bart", "homer", "lisa", "marge"]
        for name in names:
            filename_annotations = "antrenare//" + name + ".txt"
            f = open(filename_annotations)
            for line in f:
                a = line.split(os.sep)[-1]
                b = a.split(" ")
                
                image_name = "antrenare//" + name + "//" + b[0]
                bbox = [int(b[1]),int(b[2]),int(b[3]),int(b[4])]
                character = b[5].replace('\n', '')

                if character == "unknown":
                    continue

                img = cv.imread(image_name, cv.IMREAD_GRAYSCALE)
                rsz_img = cv.resize(img[bbox[1]:bbox[3], bbox[0]:bbox[2]], (36, 36))
                train_images.append(rsz_img.flatten())
                if character == "bart":
                    train_labels.append(0)
                elif character == "homer":
                    train_labels.append(1)
                elif character == "lisa":
                    train_labels.append(2)
                elif character == "marge":
                    train_labels.append(3)

        train_images = np.array(train_images)
        train_labels = np.array(train_labels).astype('int')
        
        self.svm_classifier.fit(train_images, train_labels)


    def classify(self, image):
        img = cv.resize(image, (36, 36)).flatten()
        img = img.reshape(1, -1)
        return self.svm_classifier.predict(img)
