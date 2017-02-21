import csv
import random
from random import shuffle

import cv2
import numpy as np
from sklearn.utils import shuffle as sk_shuffle

ROW, COL, CH = (160, 320, 3)  # Raw image size

NUM_BINS = 200
BIN_CAP = NUM_BINS / 5

TRIM_TOP_PIXELS = 65
TRIM_BOTTOM_PIXELS = 5

USE_SIDE_CAMERAS = True
SIDE_CAMERA_CORRECTION = 0.25

USE_FLIP_IMAGE = True

USE_ROTATE_IMAGE = False


def get_samples(directory, recovery=False):
    print("Sample directory:", directory)
    samples = []
    with open(directory + 'driving_log.csv') as file:
        reader = csv.reader(file)
        next(reader, None)  # skip the headers
        for line in reader:
            # Ignore images with no steering in recovery lap
            if not recovery or line[3] != '0':
                # Update image path relative to current directory
                line[0] = directory + 'IMG/' + line[0].split('/')[-1]
                line[1] = directory + 'IMG/' + line[1].split('/')[-1]
                line[2] = directory + 'IMG/' + line[2].split('/')[-1]
                if recovery:
                    line[3] = str(float(line[3]) / 2)
                samples.append(line)
    return samples


def random_sample_list(my_list, num_sample):
    """
    Without replacement
    If num_sample is larger than my_list size, return my_list size in random order
    """
    return [my_list[i] for i in random.sample(range(len(my_list)), min(num_sample, len(my_list)))]


def balance_samples(raw_samples):
    # Group samples into corresponding bins
    lower = np.linspace(0, 1, NUM_BINS, endpoint=False)
    upper = np.add(lower, 1. / NUM_BINS)
    sample_bins = {}
    for sample in raw_samples:
        abs_angle = abs(float(sample[3]))
        for p1, p2 in zip(lower, upper):
            if p1 <= abs_angle <= p2:
                if (p1, p2) in sample_bins:
                    sample_bins[(p1, p2)].append(sample)
                else:
                    sample_bins[(p1, p2)] = [sample]
                break

    # Cap bin size
    balanced_samples = []
    for bucket in sample_bins.values():
        balanced_samples.extend(random_sample_list(bucket, int(BIN_CAP)))
    print("cutoff size:", int(BIN_CAP))
    print("original sample size:", len(raw_samples))
    print("balanced sample size:", len(balanced_samples))
    return balanced_samples


def generator(samples, batch_size=32):
    """
    Columns: Center Image, Left Image, Right Image, Steering Angle, Throttle, Break, Speed
    Raw image size: 160x320x3
    """
    num_samples = len(samples)
    while 1:  # Used as a reference pointer so code always loops back around
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                image = None
                angle = None
                if USE_SIDE_CAMERAS:
                    rand_int = random.randint(0, 2)
                    if rand_int == 0:
                        image = cv2.imread(batch_sample[0])
                        angle = float(batch_sample[3])
                    elif rand_int == 1:
                        image = cv2.imread(batch_sample[1])
                        angle = float(batch_sample[3]) + SIDE_CAMERA_CORRECTION
                    elif rand_int == 2:
                        image = cv2.imread(batch_sample[2])
                        angle = float(batch_sample[3]) - SIDE_CAMERA_CORRECTION
                else:
                    image = cv2.imread(batch_sample[0])
                    angle = float(batch_sample[3])
                augmented_image, augmented_angle = augment_image(image, angle)
                images.append(augmented_image)
                angles.append(augmented_angle)
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sk_shuffle(X_train, y_train)


def trim_image(image):
    image = image[TRIM_TOP_PIXELS:-TRIM_BOTTOM_PIXELS, :, :]
    return cv2.resize(image, dsize=(0, 0), fx=0.5, fy=0.5)


def get_trimmed_image_size():
    return trim_image(np.zeros((ROW, COL, CH))).shape


def augment_image(image, angle):
    if USE_FLIP_IMAGE and bool(random.getrandbits(1)):
        image, angle = flip_image(image, angle)
    if USE_ROTATE_IMAGE and bool(random.getrandbits(1)):
        image, angle = rotate_image(image, angle)
    return trim_image(image), angle


def flip_image(image, angle):
    return np.fliplr(image), -angle


def rotate_image(image, angle):
    height, width, ch = image.shape
    # # [p1, p2
    # #  p3, p4]
    # # pts1 = np.float32([[0, 0], [width, 0],
    # #                    [0, height], [width, height]])
    # # pts2 = np.float32([[width * 0.1, height * -0.1], [width * 0.9, height * 0.1],
    # #                    [width * 0.1, height * 1.1], [width * 0.9, height * 0.9]])
    # pts1 = np.float32([[width * 0.1, height * 0.1], [width * 0.9, height * -0.1],
    #                    [width * 0.1, height * 0.9], [width * 0.9, height * 1.1]])
    # pts2 = np.float32([[0, 0], [width, 0],
    #                    [0, height], [width, height]])
    # M = cv2.getPerspectiveTransform(pts1, pts2)
    # return cv2.warpPerspective(image, M, (width, height)), angle
    # rotate_angle = random.randint(-3, 3)
    rotate_angle = (random.random() - 0.5) * 2
    rot_mat = cv2.getRotationMatrix2D((width / 2, height), rotate_angle, 1)
    warped_image = cv2.warpAffine(image, rot_mat, (width, height))
    rotated_image = warped_image
    # cf = abs(rotate_angle) / 50
    # shrunk_image = warped_image[int(height * cf):int(height * (1 - cf)), int(width * cf):int(width * (1 - cf))]
    # rotated_image = cv2.resize(shrunk_image, (width, height))
    return rotated_image, angle - rotate_angle / 10
