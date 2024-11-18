import numpy as np
from scipy.linalg import svd
from scipy.spatial.distance import directed_hausdorff

from affine_registration import affine_registration
from polygon_generation import generate_polygon


class RandomBasisEncoder:
    def __init__(self, basis_size=16, basis_num_vertices=6, anchors=None):
        self.basis_size = basis_size
        if anchors is None:
            anchors = [generate_polygon(center=(0, 0),
                                        avg_radius=1,
                                        irregularity=np.random.rand(),
                                        spikiness=np.random.rand(),
                                        num_vertices=basis_num_vertices) for _ in range(basis_size)]

        # self.anchors = np.array(anchors)
        self.anchors = anchors

    def encode(self, vertices):
        encoding = []
        for anchor in self.anchors:
            transformation, translation, linear_transform = affine_registration(vertices, anchor, 100, 100)

            residual_hausdorff = directed_hausdorff(transformation(vertices), anchor)[0]
            encoding += [translation[0], translation[1], linear_transform[0, 0], linear_transform[1, 1],
                         linear_transform[0, 1], linear_transform[1, 0], residual_hausdorff]

        return encoding

class RandomSVDEncoder:
    def __init__(self, basis_size=16, basis_num_vertices=6, anchors=None):
        self.basis_size = basis_size
        if anchors is None:
            anchors = [generate_polygon(center=(0, 0),
                                    avg_radius=1,
                                    irregularity=np.random.rand(),
                                    spikiness=np.random.rand(),
                                    num_vertices=basis_num_vertices) for _ in range(basis_size)]

        # self.anchors = np.array(anchors)
        self.anchors = anchors

    def _get_radian_from_matrix(self, U):
        ### To-Do: The radian may be wrong since they only consider 0-pi ###
        return np.arctan2(U[1,0], U[0,0])

    def encode(self, vertices):
        encodings = []
        for anchor in self.anchors:
            transformation, translation, linear_transform = affine_registration(vertices, anchor, 100, 100)
            U, s, Vh = svd(linear_transform)
            thetaV, thetaU = self._get_radian_from_matrix(Vh.T), self._get_radian_from_matrix(U)

            residual_hausdorff = directed_hausdorff(transformation(vertices), anchor)[0]
            encodings += [translation[0], translation[1], Vh[0, 0], Vh[0, 1], s[0], s[1], U[0, 0], U[0, 1], residual_hausdorff]
            # encodings += [translation[0], translation[1], thetaV, thetaU, s[0], s[1], residual_hausdorff]

        return encodings

class RandomRotScaleEncoder:
    def __init__(self, basis_size=16, basis_num_vertices=6, anchors=None):
        self.basis_size = basis_size
        if anchors is None:
            anchors = [generate_polygon(center=(0, 0),
                                    avg_radius=1,
                                    irregularity=np.random.rand(),
                                    spikiness=np.random.rand(),
                                    num_vertices=basis_num_vertices) for _ in range(basis_size)]

        # self.anchors = np.array(anchors)
        self.anchors = anchors

    def _get_radian_from_matrix(self, U):
        ### To-Do: The radian may be wrong since they only consider 0-pi ###
        return np.arctan2(U[1,0], U[0,0])

    def encode(self, vertices):
        encodings = []
        for anchor in self.anchors:
            transformation, translation, linear_transform = affine_registration(vertices, anchor, 100, 100)
            U, s, Vh = svd(linear_transform)
            thetaV, thetaU = self._get_radian_from_matrix(Vh.T), self._get_radian_from_matrix(U)

            residual_hausdorff = directed_hausdorff(transformation(vertices), anchor)[0]
            # encodings += [translation[0], translation[1], Vh[0, 0], Vh[0, 1], s[0], s[1], U[0, 0], U[0, 1], residual_hausdorff]
            encodings += [translation[0], translation[1], thetaV, thetaU, s[0], s[1], residual_hausdorff]

        return encodings