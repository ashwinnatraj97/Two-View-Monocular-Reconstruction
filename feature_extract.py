import numpy as np
import cv2
import matplotlib.pyplot as plt


class Feature_match(object):
    def feat_extract_match(self, img1, img2):
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        orb = cv2.ORB_create()
        key_points_1, descriptors_1 = orb.detectAndCompute(img1, None)
        key_points_2, descriptors_2 = orb.detectAndCompute(img2, None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptors_1, descriptors_2)
        matches = sorted(matches, key=lambda x: x.distance)

        matched_key_points_1 = np.asarray(
            [key_points_1[mat.queryIdx].pt for mat in matches])
        matched_key_points_2 = np.asarray(
            [key_points_2[mat.trainIdx].pt for mat in matches])

        img_matches = cv2.drawMatches(
            img1, key_points_1, img2, key_points_2, matches[:len(matches)], img2, flags=2)

        plt.imshow(img_matches)
        plt.show()

        return matched_key_points_1, matched_key_points_2


if __name__ == '__main__':
    img1 = cv2.imread('batinria0.tif')
    img2 = cv2.imread('batinria0.tif')

    feature_match_object = Feature_match()
    kp1, kp2 = feature_match_object.feat_extract_match(img1, img2)
    kp1 = kp1.astype(int)
    kp2 = kp2.astype(int)

    x1 = kp1[:, 0]
    y1 = kp1[:, 1]

    x2 = kp2[:, 0]
    y2 = kp2[:, 1]

    np.savetxt('x1.txt', x1, fmt = '%d')
    np.savetxt('y1.txt', y1, fmt = '%d')
    np.savetxt('x2.txt', x2, fmt = '%d')
    np.savetxt('y2.txt', y2, fmt = '%d')
