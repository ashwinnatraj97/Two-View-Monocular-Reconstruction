import numpy as np


class Image_Coordinates(object):
    def __init__(self):
        self.K1 = np.array([[844.310547, 0, 243.413315], [
                           0, 1202.508301, 281.529236], [0, 0, 1]])
        self.K2 = np.array([[852.721008, 0, 252.021805], [
                           0, 1215.657349, 288.587189], [0, 0, 1]])

    def image_coordinates(self, kp1, kp2):
        K1inv = np.linalg.inv(self.K1)
        K2inv = np.linalg.inv(self.K2)

        kp1 = np.transpose(kp1)
        kp2 = np.transpose(kp2)

        kp1 = np.vstack((kp1, np.ones((1, kp1.shape[1]))))
        kp2 = np.vstack((kp2, np.ones((1, kp2.shape[1]))))

        img1_pts = np.matmul(K1inv, kp1)
        img2_pts = np.matmul(K2inv, kp2)

        return np.transpose(img1_pts), np.transpose(img2_pts)


if __name__ == '__main__':
    image_coor_object = Image_Coordinates()
