import cv2
import numpy as np
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

# Distance difference between two cameras.
cam_dist = 0.055

# Get depth, gray image.
gray0  = cv2.imread("../images_sim/gray_image_0.png", cv2.IMREAD_GRAYSCALE)
gray1  = cv2.imread("../images_sim/gray_image_1.png", cv2.IMREAD_GRAYSCALE)
depth0 = np.load("../images_sim/depth_image_0.npy")
depth1 = np.load("../images_sim/depth_image_1.npy")

# Initialize IRPattern module.
ir_pattern = IRPattern(depth0.shape, intrinsic)

# Get IR pattern map from each depth image.
depth_dot0 = ir_pattern.ir_matrix_from_depth(depth0, cam_dist / 2)
depth_dot1 = ir_pattern.ir_matrix_from_depth(depth1, -cam_dist / 2)

# Apply IR pattern to gray image.
gray0[np.where(depth_dot0)] = (gray0[np.where(depth_dot0)] + 1) * 1.5
gray1[np.where(depth_dot1)] = (gray1[np.where(depth_dot1)] + 1) * 1.5

# Create depth image from two gray image.
depth = img_to_depth(gray0, gray1, intrinsic)

# Visualize.
fig = plt.figure()
ax = fig.add_subplot(221)
ax.imshow(gray0, cmap='gray')
ax = fig.add_subplot(222)
ax.imshow(gray1, cmap='gray')
ax = fig.add_subplot(223)
ax.imshow(depth, cmap='gray')
ax = fig.add_subplot(224)
ax.imshow(ir_pattern.gen_simple_ir_pattern(), cmap='gray')
plt.show()
