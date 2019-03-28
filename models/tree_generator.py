import sys
import nd.RandomGeneration as ndrg
import numpy as np

class TreeGenerator():

	def __init__(self,m,seed=None):
		self.m = m
		# first use Vitalik algorithm to obtain random hierarchy (pre-order)
		self.lbs = list(range(m))
		self.nd2d = ndrg.generate(m,labels=self.lbs,seed=seed)
		self.tree = self.createLabelTree()
		self.struct = self.get_struct()
	
	def sample_tree(self,seed=None):
		self.lbs = list(range(self.m))
		self.nd2d = ndrg.generate(self.m,labels=self.lbs,seed=seed)
		self.tree = self.createLabelTree()
		self.struct = self.get_struct()
		
		return self.struct

	# a binary tree node
	class Node: 
		def __init__(self, y): 
			self.y = y
			self.children = []
	
		def add_child(self,v):
			if len(self.children) == 0:
				child = TreeGenerator.Node(v)
				self.children.append(child)
			elif len(self.children) == 1:
				if set(v).issubset(set(self.children[0].y)):
					self.children[0].add_child(v)
				else:
					child = TreeGenerator.Node(v)
					self.children.append(child)
			else:
				if set(v).issubset(set(self.children[0].y)):
					self.children[0].add_child(v)
				else:
					self.children[1].add_child(v)
			
		def __str__(self):
			return(str(self.y))
		
	# function which transforms V node to corresponding set
	def lr_to_set(self,n):
		bin_str_rev = str(bin(n[0]+n[1]))[2:][::-1]
		ret_set = []
		for i in range(len(bin_str_rev)):
			if bin_str_rev[i]=='1':
				ret_set+=[i+1]
		return ret_set

	def createLabelTree(self):
		# first transform pvl to a list of sets
		pvl_t = []
		for el in self.nd2d:
			pvl_t.append(self.lr_to_set(el))
	
		# now that we have given a list of sets, we can easily construct our binary tree
		if len(pvl_t)==1:
			return TreeGenerator.Node(pvl_t[0])
		else:
			root = TreeGenerator.Node(pvl_t.pop(0))
			while len(pvl_t) != 0:
				root.add_child(pvl_t.pop(0))
			return root
	
	def print_tree(self):
		to_visit = [self.tree]
		while len(to_visit) != 0:
			visit_node = to_visit.pop(0)
			print(visit_node)
			to_visit = to_visit + visit_node.children[::-1]
		
	def get_struct(self):
		to_visit = [self.tree]
		return_list = []
		while len(to_visit) != 0:
			visit_node = to_visit.pop(0)
			return_list.append(visit_node.y)
			to_visit = to_visit + visit_node.children[::-1]
		return return_list






