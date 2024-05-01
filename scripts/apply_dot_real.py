import numpy as np
import cv2

from matplotlib import pyplot as plt

from utils import *
from ir_pattern import IRPattern

intrinsic = np.array([[893.82104492   ,0.             ,633.12652588],
                      [0.0            ,893.82104492   ,354.45303345],
                      [0.0            ,0.             ,1.          ]])

extrinsic = np.array([ -9.5284142039327635e-01, 2.1412216810855048e-02, 3.0271231318558661e-01, -1.0649844006838340e+00,
        1.2093728749856781e-02, 9.9939450765343596e-01, -3.2624527537913145e-02, 9.5766369044197786e-02,
        -3.0322758665373545e-01, -2.7425080553976539e-02, -9.5252343574778220e-01, 9.5766369044197786e-01,
        0., 0., 0., 1. ]).reshape(4,4)

color_intrinsic = np.array([911.3764038085938, 0.0, 649.60546875, 0.0, 910.2608032226562, 352.57177734375, 0.0, 0.0, 1.0]).reshape(3,3)

seg0 = np.load("../images_real/real_segment_image_0.npy")

depth0  = cv2.imread("../images_real/image0.png", cv2.IMREAD_GRAYSCALE)
depth1  = cv2.imread("../images_real/image1.png", cv2.IMREAD_GRAYSCALE)

depth = img_to_depth(depth0, depth1, intrinsic)

dish_img = depth * seg0

fig = plt.figure()
ax = fig.add_subplot(221)
ax.imshow(depth0, cmap='gray')
ax = fig.add_subplot(222)
ax.imshow(depth1, cmap='gray')
ax = fig.add_subplot(223)
ax.imshow(depth, cmap='gray')
ax = fig.add_subplot(224)
ax.imshow(seg0, cmap='gray')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
pcd = depth_to_pcd(depth, intrinsic)
pcd_object = pcd[np.arange(1,pcd.shape[0],200)]
ax.scatter(pcd_object[:, 0], pcd_object[:, 1], pcd_object[:, 2])
plt.show()


pcd = depth_to_pcd(dish_img, intrinsic)
pcd_object = pcd[np.arange(1,pcd.shape[0],1)]
ax.scatter(pcd_object[:, 0], pcd_object[:, 1], pcd_object[:, 2])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
pcd_w = (np.matmul(extrinsic[:3,:3], pcd_object[:,:3].T) + extrinsic[:3,3].reshape(3,1)).T
max_height = np.max(pcd_w[:,2]) - (np.max(pcd_w[:,2]) - np.min(pcd_w[:,2])) * 0.3
pcd_w = pcd_w[np.where(pcd_w[:,2] < np.max(pcd_w[:,2]) * 0.7)[0]]
min_height = np.min(pcd_w[:,2]) + (np.max(pcd_w[:,2]) - np.min(pcd_w[:,2])) * 0.15
pcd_w = pcd_w[np.where(pcd_w[:,2] > min_height)[0]]
ax.scatter(pcd_w[:, 0], pcd_w[:, 1], pcd_w[:, 2])

# ax.elev = 90
# ax.azim = 0
ax.elev = 0
ax.azim = 270
# ax.elev = 0
# ax.azim = 0
plt.show()