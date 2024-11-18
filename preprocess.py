import numpy as np
from scipy.spatial.distance import directed_hausdorff

from polygon_generation import generate_polygon
from random_polygon_basis_encoder import RandomBasisEncoder, RandomSVDEncoder, RandomRotScaleEncoder

from tqdm import tqdm

def construct_vertices(number_vertices, dataset_size=100):
    # irregs, spiks = [], []
    vs = []

    for _ in tqdm(range(dataset_size)):
        irreg, spik = np.random.rand(), np.random.rand()
        vertices = generate_polygon(center=(0, 0),
                                     avg_radius=1,
                                     irregularity=irreg,
                                     spikiness=spik,
                                     num_vertices=number_vertices)

        v = np.array(vertices)

        # irregs.append(irreg)
        # spiks.append(spik)

        vs.append(v)

    # vertices = np.array(vs)

    # return vertices
    return vs

def construct_encodings(encoder, vertices):
    dataset_size = len(vertices)
    es = []

    for i in tqdm(range(dataset_size)):
        v = vertices[i]
        e = encoder.encode(v)
        es.append(e)

    encodings = np.array(es)

    return encodings

def construct_hausdorffs(vertices):
    dataset_size = len(vertices)

    hausdorffs = np.zeros((dataset_size, dataset_size))
    for i, v1 in tqdm(enumerate(vertices), total=dataset_size):
        for j in range(i, dataset_size):
            v2 = vertices[j]

            raw_hausdorff = directed_hausdorff(v1, v2)[0]
            hausdorffs[i, j] = raw_hausdorff
            hausdorffs[j, i] = raw_hausdorff

    return hausdorffs

    # idxs = []
    # hausdorffs = []
    # for i, v1 in tqdm(enumerate(vs), total=dataset_size):
    #     random_idx = np.random.choice(dataset_size, num_cross, replace=False)
    #     for j in random_idx:
    #         v2 = vs[j]
    #
    #         idxs.append([i, j])
    #
    #         raw_hausdorff = directed_hausdorff(v1, v2)[0]
    #         hausdorffs.append(raw_hausdorff)
    #
    # return vertices, encodings, idxs, hausdorffs

data_num_vertices= 5
basis_num_vertices= 5
train_size = 10000
test_size = 1000

# print("Data Preparation:")
# print("Constructing Training Vertices...\n")
# train_vertices = construct_vertices(data_num_vertices, train_size)
# print("Constructing Training Hausdorffs...\n")
# train_targets = construct_hausdorffs(train_vertices)
#
# print("Constructing Test Vertices...\n")
# test_vertices = construct_vertices(data_num_vertices, test_size)
# print("Constructing Test Hausdorffs...\n")
# test_targets = construct_hausdorffs(test_vertices)

# print("Data Preparation:")
# print("Constructing Training Vertices...\n")
# penta_vertices = construct_vertices(5, train_size // 2)
# hexagon_vertices = construct_vertices(6, train_size // 2)
# train_vertices = penta_vertices + hexagon_vertices
# print("Constructing Training Hausdorffs...\n")
# train_targets = construct_hausdorffs(train_vertices)
#
# print("Constructing Test Vertices...\n")
# penta_vertices = construct_vertices(5, test_size // 2)
# hexagon_vertices = construct_vertices(6, test_size // 2)
# test_vertices = penta_vertices + hexagon_vertices
#
# print("Constructing Test Hausdorffs...\n")
# test_targets = construct_hausdorffs(test_vertices)

# np.savez("data/vertices-targets.npz", train_vertices=train_targets, test_vertices=test_targets, train_targets=train_targets, test_targets=test_targets)

data = np.load("dataset/de_5-be_5-et_basis-bs_32-trs_10000-tes_1000.npz")
train_vertices, train_targets, test_vertices, test_targets = data["train_vertices"], data["train_targets"], data["test_vertices"], data["test_targets"]

# all_anchors = data["basis"]
all_anchors = [generate_polygon(center=(0, 0),
                                avg_radius=1,
                                irregularity=np.random.rand(),
                                spikiness=np.random.rand(),
                                num_vertices=basis_num_vertices) for _ in range(32)]

# five_anchors = [generate_polygon(center=(0, 0),
#                                 avg_radius=1,
#                                 irregularity=np.random.rand(),
#                                 spikiness=np.random.rand(),
#                                 num_vertices=5) for _ in range(16)]
#
# six_anchors = [generate_polygon(center=(0, 0),
#                                 avg_radius=1,
#                                 irregularity=np.random.rand(),
#                                 spikiness=np.random.rand(),
#                                 num_vertices=6) for _ in range(16)]

# for encoder_type in ["basis", "svd"]:
for basis_size in [32]:
    anchors = all_anchors[:basis_size]
    # anchors = five_anchors[:basis_size//2] + six_anchors[:basis_size//2]
    for encoder_type in ["basis", "svd", "rotscale"]:
        print("Encoder type: {}, Basis size: {}".format(encoder_type, basis_size))
        if encoder_type == "basis":
            encoder = RandomBasisEncoder(basis_size, basis_num_vertices, anchors)
        elif encoder_type == "svd":
            encoder = RandomSVDEncoder(basis_size, basis_num_vertices, anchors)
        elif encoder_type == "rotscale":
            encoder = RandomRotScaleEncoder(basis_size, basis_num_vertices, anchors)
        else:
            assert False, "Unknown encoder type: {}".format(encoder_type)
        print("Constructing Training Encodings...\n")
        train_encodings = construct_encodings(encoder, train_vertices)
        print("Constructing Test Encodings...\n")
        test_encodings = construct_encodings(encoder, test_vertices)

        np.savez("dataset/de_{}-be_{}-et_{}-bs_{}-trs_{}-tes_{}".format(data_num_vertices, basis_num_vertices, encoder_type, basis_size, train_size, test_size),
                 # basis=encoder.anchors,
            # train_vertices=train_vertices,
                 train_encodings=train_encodings, train_targets=train_targets,
            # test_vertices=test_vertices,
                 test_encodings=test_encodings, test_targets=test_targets)