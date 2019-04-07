import numpy as np
import matplotlib.pyplot as plt
from math_operation import Operations
from mpl_toolkits.mplot3d import Axes3D
from open3d import *


operations_object = Operations()


def reconstruction(kp1, kp2, R, T):
    points = kp1.shape[0]
    M = np.zeros((3 * points, points + 1))

    kp1 = np.transpose(kp1)
    kp2 = np.transpose(kp2)
    kp1 = np.vstack((kp1, np.ones((1, points))))
    kp2 = np.vstack((kp2, np.ones((1, points))))
    print kp2.shape
    for i in xrange(1, points):
        x2_hat = operations_object.hat(kp2[:, i])
        a = np.matmul(np.matmul(x2_hat, R), kp1[:, i].reshape((3, 1)))
        print x2_hat.shape, R.shape, kp2[:, i].shape, a.shape
        print a

        M[3 * i - 3, i] = a[0]
        M[3 * i - 2, i] = a[1]
        M[3 * i - 1, i] = a[2]

        b = np.matmul(x2_hat, T)
        M[3 * i - 3, points] = b[0]
        M[3 * i - 2, points] = b[1]
        M[3 * i - 1, points] = b[2]

    eigenvalues, eigenvectors = np.linalg.eig(np.matmul(np.transpose(M), M))
    idx = np.argsort(eigenvalues)

    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    lamda = eigenvectors[:points, 1]
    gamma = eigenvectors[points, 1]

    if np.all(lamda >= np.zeros_like(lamda)):
        print R
        print T
        print lamda
        print gamma
        fig = plt.figure()
        ax = Axes3D(fig)
        line1 = ax.plot(kp1[0, :], kp1[1, :], np.transpose(lamda), 'ok')
        xyz = np.vstack((kp1[0:2,:], np.transpose(lamda)))
        print xyz.shape
        plt.show()
        pcd = PointCloud()
        pcd.points = Vector3dVector(np.transpose(xyz))
        write_point_cloud("test.ply", pcd)
        draw_geometries([pcd])
