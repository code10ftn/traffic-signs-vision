"""
Dataset downloaded from:
http://benchmark.ini.rub.de/?section=gtsdb&subsection=dataset

Contains OpenCV implementations using Canny edge detection and HAAR classifier.

HAAR model trained using this open source tool:
https://github.com/mrnugget/opencv-haar-classifier-training
"""

import cv2
import os
from collections import namedtuple

Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

images_dir = 'gtsdb'
classifier = cv2.CascadeClassifier(os.path.join('models-haar', 'cascade.xml'))


def show_image(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def canny_edge(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5, 5))
    image = clahe.apply(image)

    th, _ = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return cv2.Canny(image, th * 0.5, th)


def canny_method(path):
    image = cv2.imread(path)
    original_img = image.copy()

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = canny_edge(image)

    image, contours, _ = cv2.findContours(
        image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(original_img, contours, -1, (0, 0, 255), 1)

    detected = []
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(original_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        detected.append(contour)
        boxes.append(Rectangle(x, y, x + w, y + h))

    cv2.drawContours(original_img, detected, -1, (255, 0, 0), 1)
    show_image(original_img)

    return boxes


def haar_method(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    detected = classifier.detectMultiScale(image)
    boxes = []
    for (x, y, w, h) in detected:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        boxes.append(Rectangle(x, y, x + w, y + h))

    return boxes


def test(path, expected=None):
    actual = canny_method(path)
    # actual = haar_method(path)

    detected_map = {}

    for detected in actual:
        for labeled in expected:
            if overlapped_area(labeled, detected):
                if labeled not in detected_map:
                    detected_map[labeled] = 0
                detected_map[labeled] += 1

    false_negative = len(expected) - len(detected_map)
    false_positive = len(actual) - sum(detected_map.values())
    return len(detected_map), false_positive, false_negative


def overlapped_area(expected, actual):
    dx = min(expected.xmax, actual.xmax) - max(expected.xmin, actual.xmin)
    dy = min(expected.ymax, actual.ymax) - max(expected.ymin, actual.ymin)
    area = 0

    if (dx >= 0) and (dy >= 0):
        area = dx * dy

    ratio = rectangle_area(actual) / rectangle_area(expected)
    if ratio < 0.7 or ratio > 1.7:
        return False

    return area / rectangle_area(expected) > 0.8


def rectangle_area(rect):
    return (rect.ymax - rect.ymin) * (rect.xmax - rect.xmin)


def parse_labeled_data():
    image_labels = {}
    labels_path = os.path.join(images_dir, 'gt.txt')

    with open(labels_path) as f:
        content = f.readlines()
        for line in content:
            tokens = line.strip().split(';')
            if tokens[0] not in image_labels:
                image_labels[tokens[0]] = []
            image_labels[tokens[0]].append(
                Rectangle(int(tokens[1]), int(tokens[2]), int(tokens[3]), int(tokens[4])))

    return image_labels


if __name__ == '__main__':
    image_labels = parse_labeled_data()

    total_matched = 0
    total_fp = 0
    total_fn = 0

    for image in image_labels:
        src = os.path.join(images_dir, image)
        print(src)
        result = test(src, image_labels[image])
        total_matched += result[0]
        total_fp += result[1]
        total_fn += result[2]

    print('Matched: {} - FP: {} - FN: {}'.format
          (str(total_matched), str(total_fp), str(total_fn)))
    print('Recall: {}'.format
          (str(float(total_matched) / float(total_matched + total_fn))))
    print('Precision: {}'.format(
        str(float(total_matched) / float(total_matched + total_fp))))
