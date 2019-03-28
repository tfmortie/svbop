import numpy as np
from scipy.spatial import distance


def generate_ndc_split(enc, X, y):
    c = len(bin(enc)[2:])
    a = np.arange(c, dtype=int)
    rc = a[(1 << a & enc).astype(bool)]

    # leaf
    if len(rc) == 1:
        return (1 << rc[0], 0)

    # inner node
    else:
        centroids = np.zeros((len(rc), X.shape[1]))
        for i, ci in enumerate(rc):
            centroids[i] = np.mean(np.array([x for j, x in enumerate(X) if y[j] == ci]), axis=0)

        D = distance.squareform(distance.pdist(centroids))
        c_max = np.unravel_index(D.argmax(), D.shape)
        c1 = rc[c_max[0]]
        c2 = rc[c_max[1]]
        c1_group = [c1]
        c2_group = [c2]

        for ci in rc:
            if not (ci == c1 or ci == c2):
                if D[c_max[0], np.argwhere(rc == ci)] <= D[c_max[1], np.argwhere(rc == ci)]:
                    c1_group.append(ci)
                else:
                    c2_group.append(ci)

        return (np.sum(1 << np.array(c1_group)), np.sum(1 << np.array(c2_group)))


def generate(X, y):
    ds = []  # dichotomies
    s = []
    rc = np.sum([1 << i for i in np.unique(y)])
    root_split = generate_ndc_split(rc, X, y)
    s.append(root_split)
    while len(s) != 0:
        split = s.pop()
        ds.append(split)
        if split[1] != 0:
            s.append(generate_ndc_split(split[1], X, y))
            s.append(generate_ndc_split(split[0], X, y))
    return tuple(ds)

