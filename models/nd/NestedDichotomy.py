import numpy as np


# ----------------------------------------  Represents a nested dichotomy as a tree model  ----------------------------------------

class DNode:
    def __init__(self, name, left=None, right=None, split=None, model=None):
        self.name = name
        self.left = left
        self.right = right
        self.split = split
        self.p = np.ones(1)  # predicted probability for an instance
        self.depth = 0  # only for sampling comparison
        self.model = model

    def set_left(self, left):
        self.left = left

    def set_right(self, right):
        self.right = right

    def is_leaf(self):
        return self.split[1] == 0  # leafs have only one label in split

    def preorder(self, visit_fnc):
        if self is None:
            return
        visit_fnc(self)
        DNode.preorder(self.left, visit_fnc)
        DNode.preorder(self.right, visit_fnc)

    def inorder(self, visit_fnc):
        if self is None:
            return
        DNode.inorder(self.left, visit_fnc)
        visit_fnc(self)
        DNode.inorder(self.right, visit_fnc)

    def postorder(self, visit_fnc):
        if self is None:
            return
        DNode.postorder(self.left, visit_fnc)
        DNode.postorder(self.right, visit_fnc)
        visit_fnc(self)

    def __repr__(self):
        if self.left is None:
            left = ''
        else:
            left = self.left.name
        if self.right is None:
            right = ''
        else:
            right = self.right.name

        return '[' + str(self.name) + '; left=' + str(left) + '; right=' + str(right) + '; split=' + str(self.split) + '; model=' + str(self.model) + ']'


# ----------------------------------------  Parse a nested dichotomy from a list of dichotomies ----------------------------------------

def get_model_key(split):
    return str(split[0]) + '|' + str(split[1])


def parse(nd2d):
    """ Parses a ND tree from a given 2d array of dichotomies. """
    s = []  # stack for nodes
    root = DNode(get_model_key(nd2d[0]), split=nd2d[0])
    s.append(root.set_right)
    s.append(root.set_left)
    for nd in nd2d[1:]:
        link_func = s.pop()
        node = DNode(get_model_key(nd), split=nd)
        link_func(node)
        if not node.is_leaf():
            s.append(node.set_right)
            s.append(node.set_left)
    return root


# ----------------------------------------  Model training and prediction ----------------------------------------

def train_model(split, X, y, model_type, **kwargs):
    """ Trains a model for a given class split and training data. """
    data = np.zeros((X.shape[0], X.shape[1] + 2))
    data[:, :-2] = X
    data[:, -2] = y

    # select metaclass
    mc1 = data[(1 << y.astype(int) & split[0]).astype(bool)]
    mc2 = data[(1 << y.astype(int) & split[1]).astype(bool)]
    mc1.setflags(write=1)  # why is this needed? (-_-)
    mc2.setflags(write=1)  # why is this needed? (-_-)
    mc1[:, -1] = 1
    mc2[:, -1] = 2

    if len(mc1) == 0 or len(mc2) == 0:
        model = DummyModel()
    else:
        model = model_type(**kwargs)

    d = np.vstack((mc1, mc2))
    model.fit(d[:, :-2].astype(float), d[:, -1].astype(int))
    return model


def train(root, X, y, model_type, **kwargs):
    """ Trains the given nested dichotomy. """
    def train_node(node):
        if not node.is_leaf():
            node.model = train_model(node.split, X, y, model_type, **kwargs)

    root.preorder(train_node)


def predict_proba(root, X, c):
    """ Predicts the probability of instances X given the nested dichotomy and the trained models. """

    def predict(node):
        if node.is_leaf():
            assert (proba[:, int(np.log2(node.split[0]))] == 0).all()
            proba[:, int(np.log2(node.split[0]))] = node.p
        else:
            p = node.model.predict_proba(X)
            node.left.p = node.p * p[:, 0]
            node.right.p = node.p * p[:, 1]

    def reset_p(node):
        node.p = np.ones(1)

    proba = np.zeros((len(X), c))
    root.preorder(predict)
    root.preorder(reset_p)  # reset p values to default 1.0
    assert (np.abs(np.sum(proba, axis=1)-1) <= 1e-5).all(), 'Total class probability out of the precision range'
    return proba


# ----------------------------------------  Dummy model for 1-class or 0-class data ----------------------------------------

class DummyModel:
    def __init__(self):
        self.type = None

    def fit(self, X, y):
        if X.shape[0] == 0:
            self.type = 0
        elif 1 in y:
            self.type = 1
        else:
            self.type = 2

    def predict_proba(self, X):
        y = np.zeros((X.shape[0], 2), dtype=np.float)
        if self.type == 0:
            y.fill(0.5)
        elif self.type == 1:
            y[:, 0] = 1.0
        else:
            y[:, 1] = 1.0
        return y
