import cv2
import numpy as np

img_l = cv2.imread("./000020_left.jpg", cv2.IMREAD_GRAYSCALE)
img_r = cv2.imread("./000020_right.jpg", cv2.IMREAD_GRAYSCALE)

patch_size = 5
half_patch = patch_size // 2
remainder = patch_size % 2
search_pattern = 2  # one every 2 pixel

with open('./000020.txt') as file:
    line = file.readline()
    line = line.split()
    x_left = int(float(line[1]))
    x_right = int(float(line[3]))
    y_bot = int(float(line[4]))
    y_top = int(float(line[2]))
    file.close()

with open('./000020_allcalib.txt') as file:
    lines = file.readlines()
    f = float(lines[0].rstrip().split()[1])
    px = float(lines[1].rstrip().split()[1])
    py = float(lines[2].rstrip().split()[1])
    baseline = float(lines[3].rstrip().split()[1])
    file.close()

depth = np.zeros((y_bot - y_top + 1, x_right - x_left + 1))

for x in range(x_left + half_patch, x_right - half_patch):
    print("column{}".format(x))
    for y in range(y_top + half_patch, y_bot - half_patch):  # depth.shape[0]
        location = 0
        ssd = 99999
        for i in range(x_left + half_patch, x_right - half_patch, search_pattern):
            # ssd1 = 0
            # for patch_x in range(-half_patch, half_patch+1):
            #     for patch_y in range(-half_patch, half_patch+1):
            #         ssd1 += (int(img_l[y+patch_y, x+patch_x]) - int(img_r[y+patch_y, i+patch_x]))**2

            ssd1 = np.sum(np.square(
                img_l[y - half_patch:y + half_patch + remainder, x - half_patch:x + half_patch + remainder] -
                img_r[y - half_patch:y + half_patch + remainder, i - half_patch:i + half_patch + remainder]))

            if ssd1 < ssd:
                ssd = ssd1
                location = i

        if (img_l[y, x] - img_r[y, location]) == 0:
            div = 1
        else:
            div = img_l[y, x] - img_r[y, location]
        depth[y - y_top, x - x_left] = (baseline * f) / div
cv2.imwrite('depth.jpg', depth)
img = cv2.cvtColor(img_l, cv2.COLOR_GRAY2BGR)
img = cv2.rectangle(img, (x_left, y_top), (x_right, y_bot), (0, 0, 255), thickness=2)
cv2.imwrite('box.jpg', img)

half_box_x = depth.shape[1] // 2
half_box_y = depth.shape[0] // 2
z = depth[half_box_y, half_box_x]
box_center = [(half_box_x + x_left - px) * z / f, (half_box_y + y_top - py) * z / f, z]
real_pixels = np.zeros(depth.shape)

for x in range(depth.shape[1]):
    for y in range(depth.shape[0]):
        Z = depth[y, x]
        X = (x + x_left - px) * Z / f
        Y = (y + y_top - py) * Z / f
        if np.linalg.norm(np.array([box_center[0] - X, box_center[1] - Y, box_center[2] - Z])) <= 380:
            real_pixels[y, x] = img_l[y + y_top, x + x_left]
cv2.imwrite('real_pixels.jpg', real_pixels)

img1 = cv2.cvtColor(img_l, cv2.COLOR_GRAY2BGR)
img1 = cv2.rectangle(img1, (x_left, y_top), (x_right, y_bot), (0, 0, 255), thickness=2)
a = x_right - int((x_right - px) * z / f)
b = y_top - int((y_top - py) * z / f)
c = x_left - int((x_left - px) * z / f)
d = y_bot - int((y_bot - py) * z / f)
img1 = cv2.line(img1, (x_right, y_top), (a, b), (255, 0, 0), thickness=2)
img1 = cv2.line(img1, (x_left, y_bot), (c, d), (255, 0, 0), thickness=2)
img1 = cv2.line(img1, (x_left, y_top), (c, b), (255, 0, 0), thickness=2)
img1 = cv2.line(img1, (x_right, y_bot), (a, d), (255, 0, 0), thickness=2)

img1 = cv2.line(img1, (a, b), (a, d), (0, 0, 255), thickness=2)
img1 = cv2.line(img1, (c, d), (c, b), (0, 0, 255), thickness=2)
img1 = cv2.line(img1, (a, d), (c, d), (0, 0, 255), thickness=2)
img1 = cv2.line(img1, (c, b), (a, b), (0, 0, 255), thickness=2)
cv2.imwrite('3dlines.jpg', img1)
