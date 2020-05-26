import numpy as np
import cv2
from matplotlib import pyplot as plt
import cv2.xfeatures2d
import scipy.spatial


def SIFT_matching(img1, img2, threshold):
    if img1[2] is None or len(img1[2]) == 0 or img2[2] is None or len(img2[2]) == 0:
        return cv2.drawMatches(img1[0], img1[1], img2[0], img2[1], [], img2[0], flags=2)
    euclidean = scipy.spatial.distance.cdist(img1[2], img2[2], metric='euclidean')
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
    best_8 = np.zeros((8, 2))
    for i in range(len(sorted_pairs)):
        if i < 8:
            best_8[i][0] = sorted_pairs[-i][0]
            best_8[i][1] = sorted_pairs[-i][1]
        matches.append(cv2.DMatch(sorted_pairs[-i][0], sorted_pairs[-i][1], sorted_dists[-i]))
    result = cv2.drawMatches(img1[0], img1[1], img2[0], img2[1], matches[:8], img2[0], flags=2)
    result1 = cv2.drawMatches(img1[0], img1[1], img2[0], img2[1], matches, img2[0], flags=2)
    return result, result1, best_8


def SIFT_opencv(img):
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)
    result = cv2.drawKeypoints(img, kp, None)
    return [result, kp, des]


def fundamental_matrix(left, right):
    n = left.shape[0]
    A = np.zeros((n, 9))
    for i in range(n):
        A[i] = [left[i, 0] * right[i, 0], left[i, 1] * right[i, 0], right[i, 0],
              left[i, 0] * right[i, 1], left[i, 1] * right[i, 1], right[i, 1],
              left[i, 0], left[i, 1], 1]

    U, S, V = np.linalg.svd(A)
    F = V[-1].reshape(3, 3)
    U, S, V = np.linalg.svd(F)
    S[2] = 0
    F = np.dot(U, np.dot(np.diag(S), V))
    return F / F[2, 2]


def drawlines(img1,img2,lines,pts1,pts2):
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,(int(pt1[0]),int(pt1[1])),5,color,-1)
        img2 = cv2.circle(img2,(int(pt2[0]),int(pt2[1])),5,color,-1)
    return img1,img2

def findepi(best1, best2, l1, l2, fund_matrix):
    lines2 = cv2.computeCorrespondEpilines(best1.reshape(-1, 1, 2), 1, fund_matrix)
    lines2 = lines2.reshape(-1, 3)
    img3, img4 = drawlines(l2, l1, lines2, best2, best1)

    lines1 = cv2.computeCorrespondEpilines(best2.reshape(-1, 1, 2), 2, fund_matrix)
    lines1 = lines1.reshape(-1, 3)
    img5, img6 = drawlines(l1, l2, lines1, best1, best2)

    plt.subplot(121), plt.imshow(img5)
    plt.subplot(122), plt.imshow(img3)
    plt.show()

# a
l1 = cv2.imread("l1.jpg", cv2.IMREAD_GRAYSCALE)
l2 = cv2.imread("l2.jpg", cv2.IMREAD_GRAYSCALE)
l3 = cv2.imread("l3.jpg", cv2.IMREAD_GRAYSCALE)

sift1 = SIFT_opencv(l1)
sift2 = SIFT_opencv(l2)
sift3 = SIFT_opencv(l3)

result_l12, result_l120, best8 = SIFT_matching(sift1, sift2, 0.7)
result_l13, result_l130, best8_1 = SIFT_matching(sift1, sift3, 0.7)
plt.imshow(result_l12)
plt.show()
plt.imshow(result_l13)
plt.show()
# b
best1 = np.zeros((8,2))
best1_1 = np.zeros((8,2))
best2 = np.zeros((8,2))
best3 = np.zeros((8,2))

for i in range(len(best8)):
    best1[i] = sift1[1][int(best8[i][0])].pt
    best2[i] = sift2[1][int(best8[i][1])].pt
    best1_1[i] = sift1[1][int(best8_1[i][0])].pt
    best3[i] = sift3[1][int(best8_1[i][1])].pt

fund_matrix12 = fundamental_matrix(best1, best2)
fund_matrix13 = fundamental_matrix(best1_1, best3)
print("My fundamental matrix")
print(fund_matrix12)
print(fund_matrix13)
# c and d
findepi(best1, best2, l1, l2, fund_matrix12)
findepi(best1_1, best3, l1, l3, fund_matrix13)

# e
F12, mask = cv2.findFundamentalMat(best1,best2,cv2.FM_8POINT)
F13, mask = cv2.findFundamentalMat(best1_1,best3,cv2.FM_8POINT)
print("OpenCv fundamental matrix")
print(F12)
print(F13)
# f
findepi(best1, best2, l1, l2, F12)
findepi(best1_1, best3, l1, l3, F13)




