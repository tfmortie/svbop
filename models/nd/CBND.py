import numpy as np


def generate_cbnd_split(enc,  **kwargs):
    c = len(bin(enc)[2:])
    a = np.arange(c, dtype=int)
    rc = a[(1 << a & enc).astype(bool)]

    if len(rc) == 1:
        return (1 << rc[0], 0)

    cs = np.random.choice(rc, size=int(np.floor(len(rc)/2)), replace=False)
    c1_group = cs
    c2_group = [x for x in rc if x not in cs]
    return (np.sum(1 << np.array(c1_group)), np.sum(1 << np.array(c2_group)))


def generate(X, y, seed=42, **kwargs):
    ds = []  # dichotomies
    s = []
    rc = np.sum([1 << i for i in np.unique(y)])
    np.random.seed(seed)

    root_split = generate_cbnd_split(rc, **kwargs)
    s.append(root_split)
    while len(s) != 0:
        split = s.pop()
        ds.append(split)

        if split[1] != 0:
            s.append(generate_cbnd_split(split[1], **kwargs))
            s.append(generate_cbnd_split(split[0], **kwargs))
    return tuple(ds)