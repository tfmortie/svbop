import numpy as np


# ----------------------------------------  Uniform selection of binary terminally labeled trees ----------------------------------------


def utl_tree(n, seed=42):
    """ Random unrooted terminally labeled binary tree with n terminal nodes. (G. Furnas 1984) """
    edge1 = np.zeros(2 * n - 3, dtype=np.int)
    edge2 = np.zeros(2 * n - 3, dtype=np.int)
    intnod = n - 1
    e = 0
    edge1[e] = 0
    edge2[e] = 1
    np.random.seed(seed)
    for ternod in range(n)[2:]:
        ichose = np.random.randint(0, e + 1)
        e = e + 1
        edge2[e] = edge2[ichose]
        intnod = intnod + 1
        edge1[e] = intnod
        edge2[ichose] = intnod
        e = e + 1
        edge1[e] = intnod
        edge2[e] = ternod
    return edge1, edge2


def rtl_tree(n, seed=42):
    """ Random rooted terminally labeled binary tree with n terminal nodes. (G. Furnas 1984) """
    edge1, edge2 = utl_tree(n, seed)
    root_ind = 2 * n - 2
    split = np.random.randint(0, 2 * n - 3)
    edge1_new = np.zeros(root_ind, dtype=np.int)
    edge2_new = np.zeros(root_ind, dtype=np.int)

    edge1_new[:-2] = np.delete(edge1, split)
    edge1_new[-2:] = root_ind

    edge2_new[-2] = edge1[split]
    edge2_new[-1] = edge2[split]
    edge2_new[:-2] = np.delete(edge2, split)

    return edge1_new, edge2_new


# ---------------------------------------- Generating nested dichotomies ----------------------------------------

class Node:
    """ Internal class to represent a randomly generated nested dichotomy. Should be used only for ND generation. """
    def __init__(self, id, left=None, right=None):
        self.id = id
        self.label = None
        self.neighbors = []
        self.left = left
        self.right = right
        self.split = []
        self.visited = False

    def is_leaf(self):
        return len(self.neighbors) == 1

    def add_neighbor(self, node):
        self.neighbors.append(node)

    def __repr__(self):
        nbrs = ''
        for n in self.neighbors:
            nbrs = nbrs + str(n.name) + ', '
        return '[' + str(self.id) + '; left=' + str(self.left.name) + '; right=' + str(self.right.name) + '; {' + nbrs[:-2] + '}]'

    def rooted_tree(self):
        s = [self]
        while len(s) != 0:
            node = s.pop()
            node.visited = True
            if not node.is_leaf():
                for i in range(len(node.neighbors)):
                    if node.neighbors[i].visited or len(node.neighbors) == 2:  # no back links, i.e. add children only
                        s.append(node.neighbors[i - 1])
                        s.append(node.neighbors[i - 2])
                        node.left = node.neighbors[i - 1]
                        node.right = node.neighbors[i - 2]
                        break

    def preorder(self, visit_fnc):
        if self is None:
            return
        visit_fnc(self)
        Node.preorder(self.left, visit_fnc)
        Node.preorder(self.right, visit_fnc)

    def inorder(self, visit_fnc):
        if self is None:
            return
        Node.inorder(self.left, visit_fnc)
        visit_fnc(self)
        Node.inorder(self.right, visit_fnc)

    def postorder(self, visit_fnc):
        if self is None:
            return
        Node.postorder(self.left, visit_fnc)
        Node.postorder(self.right, visit_fnc)
        visit_fnc(self)


def edges_to_tree(classes, edges):
    nodes = []
    n = len(classes)
    for i in range(2 * n - 1):
        node = Node(i)
        if i < n:
            node.label = classes[i]
        nodes.append(node)
    for edge in edges:
        nodes[edge[0]].add_neighbor(nodes[edge[1]])
        nodes[edge[1]].add_neighbor(nodes[edge[0]])
    nodes[-1].rooted_tree()  # convert to a proper rooted tree
    nodes[-1].postorder(compute_splits)  # computes splits for each node
    return nodes[-1]


def compute_splits(node):
    if node.is_leaf():
        node.split = [1 << node.label, 0]
        assert node.split[0] >= 0, print('Overflow in node {}'.format(node.id))
    else:
        node.split = [sum(node.left.split), sum(node.right.split)]


def generate(n, labels=None, seed=42):
    """ Generates a random nested dichotomy for n classes. Returns a list of dichotomies in preorder. """
    ds = []  # dichotomies
    if labels is None:
        labels = list(range(n))

    def gen_nd(node):
        ds.append(tuple(node.split))

    edges = np.stack(rtl_tree(n, seed), axis=1)
    root = edges_to_tree(labels, edges)
    root.preorder(gen_nd)
    return tuple(ds)
