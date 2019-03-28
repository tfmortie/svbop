import numpy as np


def generate_bbok_split(enc):
    c = len(bin(enc)[2:])
    a = np.arange(c, dtype=int)
    rc = a[(1 << a & enc).astype(bool)]

    if len(rc) == 1:
        return (1 << rc[0], 0)

    sub_id = np.random.randint(1, np.power(2, len(rc) - 1))
    mask = format(sub_id, 'b').zfill(len(rc))
    mask = np.array(list(mask), dtype=np.int)

    c1_group = rc[mask.astype(bool)]
    c2_group = np.setdiff1d(rc, c1_group)

    return (np.sum(1 << np.array(c1_group)), np.sum(1 << np.array(c2_group)))


def generate(n, labels=None, seed=42):
    ds = []  # dichotomies
    s = []  # stack
    if labels is None:
        labels = np.arange(n, dtype=np.int)
    rc = np.sum([1 << i for i in labels])
    np.random.seed(seed)

    root_split = generate_bbok_split(rc)
    s.append(root_split)
    while len(s) != 0:
        split = s.pop()
        ds.append(split)

        if split[1] != 0:
            s.append(generate_bbok_split(split[1]))
            s.append(generate_bbok_split(split[0]))
    return tuple(ds)
