import numpy as np
import torch
import functools

HANDLED_FUNCTIONS = {}
def implements(torch_function):
	""" Register a torch function override for TensorList
	"""
	@functools.wraps(torch_function)
	def decorator(func):
		HANDLED_FUNCTIONS[torch_function] = func
		return func
	return decorator

@implements(torch.mean)
def mean(input):
	s = 0
	for x in input:
		s += torch.mean(x)
	return s/len(input)

@implements(torch.sum)
def sum(input):
	s = 0
	for x in input:
		s += torch.sum(x)
	return s

@implements(torch.cat)
def cat(tensors, dim=0):
	""" Requries tensors itself to be passed in as a TensorList
	"""
	if not all( issubclass(type(t), TensorList) for t in tensors ):
		raise NotImplementedError
	return TensorList([ \
	           torch.cat([tensors[i][j] for i in range(len(tensors))], dim=dim) \
			   for j in range(len(tensors[0]))])

@implements(torch.flatten)
def flatten(input):
	return torch.cat(list(input.flatten()))
	
class TensorList(object):
	def __init__(self, xlist):
		#if not issubclass(type(xlist), list):
			#raise ValueError("Input to TensorList() must be of subclass List")	
		self._data = xlist

	def __len__(self):
		return len(self._data)

	def __getitem__(self, key):
		if issubclass(type(key), (slice, list, tuple)):
			return TensorList(self._data[key])
		return self._data[key]

	def __setitem__(self, key, item):
		self._data[key] = item
	
	def __repr__(self):
		""" print length, shapes, and tensors
		"""
		return "TensorList(len={}, data={})".format(len(self),[x for x in self])

	def __expand__(self, obj):
		if issubclass(type(obj), (TensorList, list, tuple)) and len(obj) == len(self):
			return obj
		else:
			return [obj]*len(self)

	def __add__(self, other):
		other = self.__expand__(other)
		return TensorList([self[i] + other[i] for i in range(len(self))])
	def __radd__(self, other):
		return self + other

	def __neg__(self):
		return TensorList([-x for x in self])

	def __sub__(self, other):
		return self + (-other)
	def __rsub__(self, other):
		return other + (-self)

	def __mul__(self, other):
		other = self.__expand__(other)
		return TensorList([self[i] * other[i] for i in range(len(self))])
	def __rmul__(self, other):
		return self * other

	def __truediv__(self, other):
		other = self.__expand__(other)
		return TensorList([self[i] / other[i] for i in range(len(self))])
	def __rtruediv__(self, other):
		other = self.__expand__(other)
		return TensorList([other[i] / self[i] for i in range(len(self))])

	def __pow__(self, other):
		other = self.__expand__(other)
		return TensorList([self[i] ** other[i] for i in range(len(self))])
		
	def __getattr__(self, name):
		""" Allows operations on tensor-list to happen element-wise for all
		properties and attributes of constituent elements. 
		Example:
			- x.shape returns TensorList of constituent tensor shapes
			- x.norm(dim=(2,3)) returns TensorList of constituent tensor norms
		"""
		if callable(getattr(self[0], name)):
			def method(*args, **kwargs):
				return TensorList([getattr(x, name)(*args,**kwargs) for x in self])
			return method
		else:
			return TensorList([getattr(x, name) for x in self])

	def __setattr__(self, name, value):
		if name == "requires_grad":
			value = self.__expand__(value)
			for i in range(len(self)):
				self[i].__setattr__(name, value[i])
		else:
			super(TensorList, self).__setattr__(name, value)

	def __torch_function__(self, func, types, args=(), kwargs=None):
		if kwargs is None:
			kwargs = {}
		if not all( issubclass(t, (torch.Tensor, TensorList)) for t in types):
		   	return NotImplemented
		# element-wise operation is default if not in HANDLED_FUNCTIONS
		if func not in HANDLED_FUNCTIONS:
			args_list = [None]*len(self)
			for i in range(len(self)):
				a = [None]*len(args)
				for j in range(len(args)):
					if issubclass(type(args[j]), (TensorList, list)):
						a[j] = args[j][i]
					else:
						a[j] = args[j]
				args_list[i] = a
			return TensorList([func(*b, **kwargs) for b in args_list])
		return HANDLED_FUNCTIONS[func](*args,**kwargs)

