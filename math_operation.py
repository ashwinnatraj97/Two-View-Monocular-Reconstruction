import numpy as np


class Operations(object):
    def __init__(self):
        self.Rz1T = [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]
        self.Rz2T = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        self.R1 = None
        self.R2 = None
        self.T1 = None
        self.T2 = None
        self.U = None
        self.VT = None
        self.D = None

    def essential_matrix(self, chi):
        uchi, dchi, vchiT = np.linalg.svd(chi, full_matrices=False)
        print vchiT.shape
        E = vchiT[:, 8].reshape((3, 3))
        self.U, self.D, self.VT = np.linalg.svd(E, full_matrices=False)
        print E.shape
        if np.linalg.det(self.U) < 0 or np.linalg.det(self.VT) < 0:
            self.U, self.D, self.seVT = np.linalg.svd(-E)

        # print U.shape, D.shape, V.shape

        self.D[0] = 1
        self.D[1] = 1
        self.D[2] = 0

        E = np.matmul(np.matmul(self.U, np.diag(self.D)), self.VT)
        return E

    def rot_trans(self):
        self.R1 = np.matmul(np.matmul(self.U, self.Rz1T), self.VT)
        self.R2 = np.matmul(np.matmul(self.U, self.Rz2T), self.VT)

        T_hat1 = np.matmul(
            np.matmul(np.matmul(self.U, self.Rz1T), np.diag(self.D)), np.transpose(self.U))
        T_hat2 = np.matmul(
            np.matmul(np.matmul(self.U, self.Rz2T), np.diag(self.D)), np.transpose(self.U))

        self.T1 = np.array([[-T_hat1[1, 2]], [T_hat1[0, 2]], [-T_hat1[0, 1]]])
        self.T2 = np.array([[-T_hat2[1, 2]], [T_hat2[0, 2]], [-T_hat2[0, 1]]])

        return self.R1, self.R2, self.T1, self.T2

    def hat(self, v):
        return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
