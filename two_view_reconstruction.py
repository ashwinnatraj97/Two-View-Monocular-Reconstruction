import cv2
import numpy as np
import matplotlib.pyplot as plt
from feature_extract import Feature_match
from image_coor import Image_Coordinates
from math_operation import Operations
from Reconstruction import reconstruction


if __name__ == '__main__':
    img1 = cv2.imread('batinria0.tif')
    img2 = cv2.imread('batinria0.tif')

    feature_match_object = Feature_match()
    image_coordinates_object = Image_Coordinates()
    operations_object = Operations()

    kp1, kp2 = feature_match_object.feat_extract_match(img1, img2)
    # print kp1.shape, kp2.shape
    img1pts, img2pts = image_coordinates_object.image_coordinates(kp1, kp2)

    print img1pts.shape, img2pts.shape

    # Row Wise Kronecker Prodcuct computation
    chi = np.einsum('ij, ik->ijk', img1pts,
                    img2pts).reshape(img1pts.shape[0], -1)
    rank = np.linalg.matrix_rank(chi)

    E = operations_object.essential_matrix(chi)
    R1, R2, T1, T2 = operations_object.rot_trans()
    reconstruction(kp1, kp2, R1, T1)
    reconstruction(kp1, kp2, R1, T2)
    reconstruction(kp1, kp2, R2, T1)
    reconstruction(kp1, kp2, R2, T2)
