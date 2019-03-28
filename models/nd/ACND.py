import numpy as np
import numpy.ma as ma
from scipy.spatial import distance
from NestedDichotomies.nd import NestedDichotomy as nd


def single(u, v):
    y = distance.cdist(u, v)
    return np.min(y)


def complete(u, v):
    y = distance.cdist(u, v)
    return np.max(y)


def average(u, v):
    y = distance.cdist(u, v)
    return np.sum(y) / (len(u) * len(v))


def generate(X, y, method=average):
    """ Creates a cluster structure agglomeratively from predefinded clusters """
    #c = len(np.unique(y))
    pc = [np.array([x for i, x in enumerate(X) if y[i] == j]) for j in np.unique(y)]  # pre-cluster containing single classes
    labels = 1 << np.unique(y)  # np.arange(c, dtype=int)
    nodes = {}
    len_pc = len(pc)
    dist = np.zeros((len_pc, len_pc)) + np.iinfo(np.int32).max
    for i in np.arange(len_pc - 1):
        for j in np.arange(len_pc - 1 - i) + i + 1:
            dist[i, j] = method(pc[i], pc[j])

    mask = np.zeros(dist.shape, dtype=bool)  # 'removed' elements
    rdist = ma.masked_array(dist, mask=mask)
    pcl = np.array([len(p) for p in pc])

    while len(ma.compressed(rdist)) != 1:
        rdist = ma.masked_array(dist, mask=mask)
        rdist._sharedmask = False  # why?
        rlabels = ma.masked_array(labels, mask=mask[0])
        rlabels._sharedmask = False  # why?
        rpcl = ma.masked_array(pcl, mask=mask[0])
        rpcl._sharedmask = False  # why?

        min_ind = sorted(np.unravel_index(rdist.argmin(), rdist.shape))  # clusters with the smallest distance
        split = [rlabels[min_ind[0]], rlabels[min_ind[1]]]

        if not split[0] & split[0] - 1:  # leaf
            split_leaf = [split[0], 0]
            nodes[np.sum(split_leaf)] = nd.DNode(nd.get_model_key(split_leaf), split=split_leaf)

        if not split[1] & split[1] - 1:  # leaf
            split_leaf = [split[1], 0]
            nodes[np.sum(split_leaf)] = nd.DNode(nd.get_model_key(split_leaf), split=split_leaf)

        nodes[np.sum(split)] = nd.DNode(nd.get_model_key(split), split=split)

        #  update distance matrix and labels
        rdist[min_ind[0]] = (rdist[min_ind[0]] * rpcl[min_ind[0]] + rdist[min_ind[1]] * rpcl[min_ind[1]]) / (rpcl[min_ind[0]] + rpcl[min_ind[1]])
        rlabels[min_ind[0]] = rlabels[min_ind[0]] + rlabels[min_ind[1]]
        rpcl[min_ind[0]] = rpcl[min_ind[0]] + rpcl[min_ind[1]]

        mask[min_ind[1]] = True
        mask[:, min_ind[1]] = True

    # generate the corresponding nested dichotomy
    for n in nodes.values():
        if not n.is_leaf():
            n.left = nodes[n.split[0]]
            n.right = nodes[n.split[1]]

    root = nodes[np.sum([1 << i for i in np.unique(y)])]
    ds = []  # dichotomies

    def gen_nd(node):
        ds.append(tuple(node.split))

    root.preorder(gen_nd)
    return tuple(ds)
