from data_processor import *
import pickle

DATA_DIR = './large_files/sample/'  # Sample data from Udacity
DATA_DIR3 = './large_files/train3/'  # Short smooth driving
DATA_DIR4 = './large_files/train4/'  # Full lap smooth driving (2 minutes)
DATA_DIR5 = './large_files/train5/'  # Recovery lap
DATA_DIR6 = './large_files/train6/'  # Hard corner smooth driving
DATA_DIR7 = './large_files/train7/'  # 20 minutes smooth driving
DATA_DIR8 = './large_files/train8/'  # The split road correction
DATA_DIR9 = './large_files/train9/'  # 25 minutes smooth driving (slightly emphasize on hard corners)

"""Visualize"""
# sample = get_samples()[0]
# image = cv2.imread(sample[0])
# angle = float(sample[3])
# print(angle)
# cv2.imshow('test', image)
# cv2.waitKey(0)
# rotated_image, rotated_angle = rotate_image(image, angle)
# print(rotated_angle)
# cv2.imshow('rotated', rotated_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# exit()


"""Steering Angle Distribution"""
# import matplotlib.pyplot as plt
#
# plt.figure(1)
#
# raw_samples = get_samples(DATA_DIR4) + get_samples(DATA_DIR7) + get_samples(DATA_DIR5, True)
# angles = np.array([abs(float(sample[3])) for sample in raw_samples])
# plt.subplot(211)
# plt.title("Before")
# plt.hist(angles, bins=np.linspace(0, 1, NUM_BINS, endpoint=False))
#
# angles = np.array([abs(float(sample[3])) for sample in balance_samples(raw_samples)])
# plt.subplot(212)
# plt.title("After")
# plt.hist(angles, bins=np.linspace(0, 1, NUM_BINS, endpoint=False))
# plt.savefig('distribution.png', bbox_inches='tight')


"""Flip image"""
# import matplotlib.pyplot as plt
#
# plt.axis('off')
# fig = plt.figure(1)
#
#
# samples = get_samples(DATA_DIR5, True)
# index = random.randint(0, len(samples))
# image_rgb = cv2.imread(samples[index][0])
# image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
# angle = float(samples[index][3])
# fig.add_subplot(1, 2, 1)
# plt.title("Steering Angle:" + str(angle))
# plt.imshow(image)
#
# image, angle = flip_image(image, angle)
# fig.add_subplot(1, 2, 2)
# plt.title("Steering Angle:" + str(angle))
# plt.imshow(image)
# axes = fig.get_axes()
# for axis in axes:
#     axis.set_axis_off()
# plt.savefig('test.png', bbox_inches='tight')


"""Test running saved model"""
# model = load_model('./model_temp.h5')
#
#
# def get_angle(image):
#     image_array = np.asarray(image)
#     angle = model.predict(image_array[None, 80:, :, :], batch_size=1)
#     return angle
#
#
# DATA_DIR = './large_files/sample/'
#
# samples = []
# with open(DATA_DIR + 'driving_log.csv') as file:
#     reader = csv.reader(file)
#     next(reader, None)  # skip the headers
#     for line in reader:
#         samples.append(line)
#
# for i in range(100):
#     name = DATA_DIR + 'IMG/' + samples[random.randint(0, len(samples))][0].split('/')[-1]
#     center_image = cv2.imread(name)
#     print(get_angle(center_image))


"""Video from Images"""
# import matplotlib.pyplot as plt
# raw_samples = get_samples(DATA_DIR4)
#
# plot = None
# for sample in raw_samples:
#     image_rgb = cv2.imread(sample[0])
#     image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
#     if plot is None:
#         plt.axis('off')
#         plot = plt.imshow(cv2.imread(raw_samples[0][0]))
#     else:
#         plot.set_data(image)
#     plt.pause(.00001)
#     plt.draw()


"""Plot training/validation loss"""
import matplotlib.pyplot as plt

with open('history.pickle', 'rb') as hist_in:
    history = pickle.load(hist_in)

plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.xlabel("epochs")
plt.ylabel("RMS")
plt.savefig('test.png', bbox_inches='tight')

"""Others"""
# raw_samples = get_samples(DATA_DIR4) + get_samples(DATA_DIR7)
# samples = balance_samples(raw_samples)
# shuffle(samples)
# image = cv2.imread(samples[0][1])
# cv2.imshow('image', image)
# cv2.waitKey(0)
# image = trim_image(image)
# cv2.imshow('trimmed', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
