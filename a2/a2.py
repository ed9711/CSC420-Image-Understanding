import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
import sklearn.feature_extraction.image as skl
import cv2.xfeatures2d
import scipy


def display_image(img, file_name=None):
    flt_img = img.astype(float)
    img_max, img_min = np.max(flt_img), np.min(flt_img)

    norm_img = (((flt_img - img_min) / (img_max - img_min)) * 255).astype(np.uint8)

    if len(img.shape) == 2:
        plt.imshow(norm_img, cmap='gray')
    elif len(img.shape) == 3:
        plt.imshow(cv2.cvtColor(norm_img, cv2.COLOR_BGR2RGB))
    plt.show()


def linear_interpolation_twice(img, size):
    shape = img.shape
    result = np.ones((int(shape[0] * size), int(shape[1] * size), shape[2]))
    for i in range(shape[0] * size):
        for j in range(shape[1] * size):
            if j % size == 0:
                if i / size < shape[0] and j / size < shape[1]:
                    result[i][j] = img[int(i / size)][int(j / size)]
            else:
                if math.ceil(i / size) < shape[0] and math.ceil(j / size) < shape[1]:
                    result[i][j] = ((math.ceil(j / size) - j / size) / (
                            math.ceil(j / size) - math.floor(j / size))) * \
                                   img[math.floor(i / size)][math.floor(j / size)] + (
                                           (j / size - math.floor(j / size)) / (
                                           math.ceil(j / size) - math.floor(j / size))) * \
                                   img[math.ceil(i / size)][math.ceil(j / size)]
                else:
                    result[i][j] = img[math.floor(i / size)][math.floor(j / size)]
    # except:
    #     print(i / size, shape[0])
    #     print(j / size, shape[1])
    return result


def linear_interpolation(img, size):
    shape = img.shape
    result = np.ones((int(shape[0]), int(shape[1] * size), shape[2]))
    for i in range(shape[0]):
        for j in range(shape[1] * size):
            if j % size == 0:
                result[i][j] = img[int(i)][int(j / size)]
            else:
                if math.ceil(i) < shape[0] and math.ceil(j / size) < shape[1]:
                    result[i][j] = ((math.ceil(j / size) - j / size) / (math.ceil(j / size) - math.floor(j / size))) * \
                                   img[i][math.floor(j / size)] + ((j / size - math.floor(j / size)) / (
                            math.ceil(j / size) - math.floor(j / size))) * img[i][math.ceil(j / size)]
                else:
                    result[i][j] = img[i][math.floor(j / size)]
    print("here")
    shape = result.shape
    result1 = np.ones((int(shape[0] * size), int(shape[1]), shape[2]))
    for i in range(shape[0] * size):
        for j in range(shape[1]):
            if i % size == 0:
                result1[i][j] = result[int(i / size)][j]
            else:
                if math.ceil(i / size) < shape[0] and math.ceil(j) < shape[1]:

                    result1[i][j] = ((math.ceil(i / size) - i / size) / (math.ceil(i / size) - math.floor(i / size))) * \
                                    result[math.floor(i / size)][j] + ((i / size - math.floor(i / size)) / (
                            math.ceil(i / size) - math.floor(i / size))) * result[math.ceil(i / size)][j]
                else:
                    result1[i][j] = result[math.floor(i / size)][j]

    return result1


def corner(img, a, key):
    blur = cv2.GaussianBlur(img, (5, 5), 7)
    Ix = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=5)
    Iy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=5)
    IxIy = np.multiply(Ix, Iy)
    Ix2 = np.multiply(Ix, Ix)
    Iy2 = np.multiply(Iy, Iy)
    Ix2_blur = cv2.GaussianBlur(Ix2, (7, 7), 10)
    Iy2_blur = cv2.GaussianBlur(Iy2, (7, 7), 10)
    IxIy_blur = cv2.GaussianBlur(IxIy, (7, 7), 10)
    det = np.multiply(Ix2_blur, Iy2_blur) - np.multiply(IxIy_blur, IxIy_blur)
    trace = Ix2_blur + Iy2_blur
    if key == "harris":
        R = det - a * np.multiply(trace, trace)
    elif key == "brown":
        with np.errstate(divide='ignore', invalid='ignore'):
            R = det / trace
    return R


def corner_eigen(img):
    print(img.dtype)
    eig = cv2.cornerEigenValsAndVecs(img, 7, 7)
    result = np.empty(img.shape, dtype=np.float32)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            lambda_1 = eig[i, j, 0]
            lambda_2 = eig[i, j, 1]
            result[i, j] = lambda_1 * lambda_2 - 0.05 * pow((lambda_1 + lambda_2), 2)
    return result


def patch_view(img, patch_h, patch_w):
    h, w = np.array(img.shape) - np.array([patch_h, patch_w]) + 1
    return np.lib.stride_tricks.as_strided(np.ascontiguousarray(img), shape=(h, w, patch_h, patch_w),
                                           strides=img.strides + img.strides, writeable=False)


def suppression(img, neighbours):
    img_pad = np.pad(img, ((1, 1), (1, 1)), mode='constant', constant_values=0)
    neighbours_pad = []
    for item in neighbours:
        neighbours_pad.append(np.pad(item, ((1, 1), (1, 1)), mode='constant', constant_values=0))

    # img_patches = skl.extract_patches_2d(img_pad, (3, 3), min(h,w))

    img_patches = patch_view(img_pad, 3, 3)
    neighbours_patches = []
    for item in neighbours_pad:
        neighbours_patches.append(patch_view(item, 3, 3))

    patch_max = np.amax(img_patches, axis=(2, 3))

    neighbours_patches_maxes = []
    for item in neighbours_patches:
        neighbours_patches_maxes.append(np.amax(item, axis=(2, 3)))
    all_max = np.amax(np.dstack([patch_max] + neighbours_patches_maxes), axis=2)

    is_cur_max = np.equal(img, all_max)

    return img * is_cur_max


def find_max(img, length, sigma):
    scale = 2 ** (1 / length)
    lapace = []
    blur = []
    for i in range(length):
        blur_width = sigma * (scale ** i)
        lapace.append(cv2.GaussianBlur(img, (5, 5), sigma))
        blur.append(blur_width)

    counter = 0
    for item in lapace:
        lap = np.abs(cv2.Laplacian(item, ddepth=cv2.CV_32F, ksize=5, scale=1))
        print(item)
        lapace[counter] = lap
        counter += 1
    suppressed = [suppression(lapace[0], [lapace[1]])]
    for i in range(len(lapace) - 2):
        (down, cur, up) = lapace[i:i + 3]
        suppressed += ([suppression(cur, [down, up])])
    suppressed += ([suppression(lapace[-1], [lapace[-2]])])

    stacked = np.dstack(suppressed)
    max_id = np.argmax(stacked, axis=2)
    max_item = np.amax(stacked, axis=2)
    blur = np.array(blur)
    corr_blur = blur[max_id]

    max_coords = np.nonzero(max_item)
    max_blur = corr_blur[max_coords]
    responses = max_item[max_coords]
    return np.array([responses, *max_coords, max_blur]).transpose()


def SIFT(img, length, sigma, threshold):
    result = np.dstack([np.copy(img)] * 3)
    shortest_side = min(img.shape[0], img.shape[1])
    reductions = int(np.log2(shortest_side)) - 1
    cur_image = img
    pyramids = []

    for i in range(reductions):
        scale = 2 ** i
        pyramids.append([cur_image, scale])
        cur_image = cv2.pyrDown(cur_image)

    max = []

    for img, scale in pyramids:
        oct_max = find_max(img, length, sigma)
        oct_max[:, 1:] *= scale
        max += [oct_max]
    max = np.concatenate(max)
    threshold1 = np.percentile(max[:, 0], threshold)
    for (response, y, x, blur_width) in max:
        if response > threshold1:
            radius = blur_width * (2 ** 0.5)
            cv2.circle(result, (int(x), int(y)), int(radius), (0, 255, 0), 2)

    return result


def SIFT_opencv(img):
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)
    result = cv2.drawKeypoints(img, kp, None)
    return [result, kp, des]


def SIFT_matching(img1, img2, threshold):
    if img1[2] is None or len(img1[2]) == 0 or img2[2] is None or len(img2[2]) == 0:
        return cv2.drawMatches(img1[0], img1[1], img2[0], img2[1], [], img2[0], flags=2)
    euclidean = scipy.spatial.distance.cdist(img1[2], img2[2], metric='euclidean')
    # it is possible that this step might run out of space, if so, need more memory
    sorted1 = np.argsort(euclidean, axis=1)
    closest, closest1 = sorted1[:, 0], sorted1[:, 1]
    left_id = np.arange(img1[2].shape[0])
    dist_ratios = euclidean[left_id, closest] / euclidean[left_id, closest1]
    suppressed = dist_ratios * (dist_ratios < threshold)
    left_id = np.nonzero(suppressed)[0]
    right_id = closest[left_id]
    pairs = np.stack((left_id, right_id)).transpose()
    pair_dists = euclidean[pairs[:, 0], pairs[:, 1]]
    sorted_dist_id = np.argsort(pair_dists)
    sorted_pairs = pairs[sorted_dist_id]
    sorted_dists = pair_dists[sorted_dist_id].reshape((sorted_pairs.shape[0], 1))

    matches = []
    print("Best 10 values: ")
    for i in range(10):
        print("Pairs: {}, dist: {}".format(sorted_pairs[-i], sorted_dists[-i]))
        matches.append(cv2.DMatch(sorted_pairs[-i][0], sorted_pairs[-i][1], sorted_dists[-i]))
    result = cv2.drawMatches(img1[0], img1[1], img2[0], img2[1], matches, img2[0], flags=2)
    return result


def add_noise(img):
    noise = np.random.normal(0, 0.08, img.shape[:2])
    result = img + noise
    result = result.astype(np.uint8)
    return result


# image = cv2.imread("NASDAQ.AAL.jpg")
# image = image.astype(np.float32) / 255
# image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image1 = cv2.imread("NASDAQ.AAL.jpg", cv2.IMREAD_GRAYSCALE).astype(np.float32)
# sample1 = cv2.imread("sample1.jpg", cv2.IMREAD_GRAYSCALE)
# sample2 = cv2.imread("sample2.jpg", cv2.IMREAD_GRAYSCALE)
col1 = cv2.imread("colourSearch.png")
col2 = cv2.imread("colourTemplate.png")
# plt.imshow(col1)
# plt.show()
# plt.imshow(col2)
# plt.show()

# cv2.imshow('image', image1)
# cv2.waitKey(0)
# display_image(image1)

# Q1
# result1 = linear_interpolation_twice(image1, 4)
# result = linear_interpolation(image1, 4)
# result = np.round(result * 255).astype(np.uint8)
# # result1 = np.round(result1 * 255).astype(np.uint8)
# plt.imshow(result)
# plt.show()

# Q2
# result = corner(image1, 0.05, "harris")
# result1 = corner(image2, 0.05, "harris")
# result1 = np.round(result1 * 255).astype(np.uint8)
#
# # result = corner_eigen(image1)
# result = np.round(result * 255).astype(np.uint8)
#
# plt.imshow(result1)
# plt.show()



# Q3
result = SIFT(image1, 5, 9, 98)
result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
print(image1.shape)
plt.imshow(image1+result)
plt.show()

# Q4


# sample1 = cv2.normalize(sample1, sample1, 0, 1, cv2.NORM_MINMAX)
# sample2 = cv2.normalize(sample2, sample2, 0, 1, cv2.NORM_MINMAX)
#
# sample1 = add_noise(sample1)
# sample2 = add_noise(sample2)
#
# result = SIFT_opencv(sample1*255)
# result1 = SIFT_opencv(sample2*255)
# plt.imshow(result[0])
# plt.show()
# plt.imshow(result1[0])
# plt.show()
# result2 = SIFT_matching(result, result1, 0.8)

# b, g, r = cv2.split(col1)
# b1, g1, r1 = cv2.split(col2)
# result = [SIFT_opencv(item) for item in [b, g, r]]
# result1 = [SIFT_opencv(item) for item in [b1, g1, r1]]
#
# result2 = [SIFT_matching(result[i], result1[i], 0.8) for i in range(3)]
# for i in range(3):
#     result2[i] = cv2.cvtColor(result2[i], cv2.COLOR_BGR2GRAY)
# result3 = cv2.merge(result2)
#
# plt.imshow(result3)
# plt.show()
#
