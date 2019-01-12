"""
Dataset downloaded from:
http://benchmark.ini.rub.de/?section=gtsdb&subsection=dataset

Prepares data for training a Faster R-CNN detection model following this tutorial:
https://docs.microsoft.com/en-us/cognitive-toolkit/Object-Detection-using-Faster-R-CNN
"""

import os
import sys
from collections import namedtuple
from random import shuffle
from shutil import copyfile
from PIL import Image

BoundingBox = namedtuple('BoundingBox', ['x1', 'y1', 'x2', 'y2', 'id'])


def ppm2jpg(directory):
    for dirpath, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith('.ppm'):
                ppm_path = os.path.join(directory, dirpath, filename)
                image = Image.open(ppm_path)
                jpg_path = ppm_path.replace('ppm', 'jpg')
                image.save(jpg_path)
                os.remove(ppm_path)


def load_class_ids(directory):
    class_ids_path = os.path.join(directory, 'classes.txt')
    class_ids = {}

    with open(class_ids_path) as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split('=')
            class_ids[parts[0]] = parts[1]

    return class_ids


def gtsdb2cntk(directory):
    class_ids = load_class_ids(directory)

    gt_path = os.path.join(directory, 'gt.txt')
    annotations = {}

    with open(gt_path) as gt:
        lines = gt.readlines()
        for line in lines:
            parts = line.strip().split(';')
            img = parts[0].split('.')[0]
            if img not in annotations:
                annotations[img] = []
            annotations[img].append(BoundingBox(*parts[1:]))

    images_path = os.path.join(directory, 'Images')

    for img in annotations:
        bboxes_path = os.path.join(images_path, img + '.bboxes.tsv')
        labels_path = os.path.join(images_path, img + '.bboxes.labels.tsv')
        with open(bboxes_path, 'w') as bboxes, open(labels_path, 'w') as labels:
            for b in annotations[img]:
                bboxes.write('{}\t{}\t{}\t{}\n'.format(b.x1, b.y1, b.x2, b.y2))
                labels.write(class_ids[b.id] + '\n')


def move_images(images, src, dst):
    for img in images:
        jpg_name = img + '.jpg'
        copyfile(os.path.join(src, jpg_name),
                 os.path.join(dst, jpg_name))

        bboxes_name = img + '.bboxes.tsv'
        copyfile(os.path.join(src, bboxes_name),
                 os.path.join(dst, bboxes_name))

        labels_name = img + '.bboxes.labels.tsv'
        copyfile(os.path.join(src, labels_name),
                 os.path.join(dst, labels_name))


def split_dataset(src, dst):
    gt_path = os.path.join(src, 'gt.txt')
    images = []

    with open(gt_path) as gt:
        lines = gt.readlines()
        for line in lines:
            img = line.strip().split('.')[0]
            if img not in images:
                images.append(img)

    shuffle(images)

    n = len(images)
    n_train = int(n*0.7)
    n_test = int(n*0.9)

    train = images[:n_train]
    validation = images[n_train:n_test]
    test = images[n_test:]

    src = os.path.join(src, 'Images')

    move_images(train, src, os.path.join(dst, 'positive'))
    move_images(validation, src, os.path.join(dst, 'testImages'))
    move_images(test, src, os.path.join(dst, 'testFinal'))


if __name__ == '__main__':
    directory = sys.argv[1]
    # ppm2jpg(directory)
    # gtsdb2cntk(directory)
    # split_dataset(directory, sys.argv[2])
