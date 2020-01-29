"""
Random hierarchy generator with arbitrary branching degree
"""

import numpy as np

class TreeGenerator():
    def __init__(self,m):
        self.m = m

    def split(self,y,max_splits=1):
        if max_splits < 1 or len(y) == 1:
            return [y]
        elif len(y) == 2:
            return [[y[0]],[y[1]]]
        else:
            min_i, max_i = 1, len(y)
            split_i = np.random.randint(min_i,max_i)
            to_split = np.random.randint(2)
            if to_split==0: 
                return self.split(y[:split_i],max_splits-1) + [y[split_i:]]
            else: 
                return [y[:split_i]] + self.split(y[split_i:],max_splits-1)

    def GenerateHierarchy(self,m_s=1):
        to_split_q = [list(range(1,self.m+1))]
        ret_list = [list(range(1,self.m+1))]
        while len(to_split_q)!=0:
            node = to_split_q.pop(0)
            chunks = self.split(node,max_splits=np.random.randint(1,m_s+1))
            ret_list.extend(chunks)
            to_split_q.extend([x for x in chunks if len(x)>1])
        return ret_list