"""
Dataset downloaded from:
http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset

Prepares data for training a ResNet recognition model following this tutorial:
https://docs.microsoft.com/en-us/cognitive-toolkit/Hands-On-Labs-Image-Recognition
"""

import csv
import os
import random

train_set_path = os.path.join('gtsrb', 'data', 'train')

test_labels_csv = os.path.join('gtsrb', 'GTSRB_Final_Test_Images', 'GTSRB',
                               'Final_Test', 'Images', 'GT-final_test.csv')

train_labels_csv_root = os.path.join('gtsrb', 'GTSRB_Final_Training_Images',
                                     'GTSRB', 'Final_Training', 'Images')

test_map_txt = os.path.join('gtsrb', 'test_map.txt')

train_map_txt = os.path.join('gtsrb', 'train_map.txt')


def move_train_images_up():
    classes = [x[0] for x in os.walk(train_set_path)]
    classes.pop(0)  # skip root folder

    for directory in classes:
        class_id = directory.split(os.path.sep)[3]
        print(class_id)

        for file in os.listdir(directory):
            image_name = os.fsdecode(file)

            if not image_name.endswith('.ppm'):
                continue

            prefixed_image_name = class_id + '_' + image_name
            src = os.path.join(directory, image_name)
            dst = os.path.join(train_set_path, prefixed_image_name)
            os.rename(src, dst)


def make_test_map():
    prefix = os.path.join('gtsrb', 'data', 'test') + os.path.sep

    with open(test_labels_csv, 'rt') as csvfile, open(test_map_txt, 'w') as test_map:
        test_labels = csv.reader(csvfile)

        for row in test_labels:
            parts = row[0].split(';')
            image_name = parts[0]
            class_id = parts[7]
            map_row = prefix + image_name + '\t' + class_id + '\n'
            if 'Filename' not in map_row:
                test_map.write(map_row)


def make_train_map():
    prefix = os.path.join('gtsrb', 'data', 'train') + os.path.sep

    classes = [x[0] for x in os.walk(train_labels_csv_root)]
    classes.pop(0)  # skip root folder

    with open(train_map_txt, 'w') as train_map:
        for directory in classes:
            class_id_long = directory.split(os.path.sep)[5]
            class_csv_name = 'GT-' + class_id_long + '.csv'
            class_csv_path = os.path.join(directory, class_csv_name)

            with open(class_csv_path, 'rt') as csvfile:
                train_labels = csv.reader(csvfile)

                for row in train_labels:
                    parts = row[0].split(';')
                    image_name = parts[0]
                    class_id = parts[7]
                    map_row = prefix + class_id_long + '_' + image_name + '\t' + class_id + '\n'
                    if 'Filename' not in map_row:
                        train_map.write(map_row)


def shuffle_train_map():
    lines = open(train_map_txt).readlines()
    random.shuffle(lines)
    open(train_map_txt, 'w').writelines(lines)


if __name__ == '__main__':
    # move_train_images_up()
    # make_test_map()
    # make_train_map()
    # shuffle_train_map()
