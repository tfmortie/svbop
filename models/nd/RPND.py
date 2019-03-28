import numpy as np
from NestedDichotomies.nd import NestedDichotomy


def generate_rpnd_split(enc, X, y, model_type,  **kwargs):
    c = len(bin(enc)[2:])
    a = np.arange(c, dtype=int)
    rc = a[(1 << a & enc).astype(bool)]

    if len(rc) == 1:
        return (1 << rc[0], 0)

    cs = np.random.choice(rc, size=2, replace=False)
    c1_group = [cs[0]]
    c2_group = [cs[1]]
    split = [1 << cs[0], 1 << cs[1]]
    model = NestedDichotomy.train_model(split, X, y, model_type, **kwargs)
    rc = set(rc)
    rc.remove(cs[0])
    rc.remove(cs[1])

    for ci in rc:
        mc = np.array([x for i, x in enumerate(X) if y[i] == ci])
        c = model.predict(mc)
        if (c == 1).sum() >= len(c) / 2:
            c1_group.append(ci)
        else:
            c2_group.append(ci)
    return (np.sum(1 << np.array(c1_group)), np.sum(1 << np.array(c2_group)))


def generate(X, y, model_type, seed=42, **kwargs):
    ds = []  # dichotomies
    s = []
    rc = np.sum([1 << i for i in np.unique(y)])
    np.random.seed(seed)

    root_split = generate_rpnd_split(rc, X, y, model_type, **kwargs)
    s.append(root_split)
    while len(s) != 0:
        split = s.pop()
        ds.append(split)

        if split[1] != 0:
            s.append(generate_rpnd_split(split[1], X, y, model_type, **kwargs))
            s.append(generate_rpnd_split(split[0], X, y, model_type, **kwargs))
    return tuple(ds)


