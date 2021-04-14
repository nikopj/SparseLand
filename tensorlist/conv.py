import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorlist as tl

def adjoint(A):
	""" Adjoint operator for conv kernel
	"""
	return A.transpose(0,1).flip(2,3)

def upsample(x, m=2, offset=0):
	""" Zero-filling operator
	"""
	if m==1:
		return x
	v = torch.zeros((x.shape[0], x.shape[1], m*x.shape[2], m*x.shape[3]), dtype=x.dtype, device=x.device)
	v[:, :, offset::m, offset::m]  = x
	return v

class Conv2d(nn.Module):
	""" Convolution with down-sampling.
	nic: num. input channels
	noc: num. output channels
	ks : kernel size (square kernels only)
	stride: down-sampling factor (decimation)
	pad_mode: type of padding before convolution.
	c_groups: if true, performs separate convolutions on nic-sized channel groups of input.
	"""
	def __init__(self, nic, noc, ks, stride=1,pad_mode='constant',dilation=1,c_groups=False,transposed=False):
		super(Conv2d, self).__init__()
		if transposed:
			self.forward = self._adjoint
			self.adjoint = self._forward
			temp = nic; nic = noc; noc = temp
		else:
			self.forward = self._forward
			self.adjoint = self._adjoint
		weight = torch.randn(noc,nic,ks,ks) 
		weight = weight/torch.norm(weight, dim=(2,3), keepdim=True)
		self._weight = nn.Parameter(weight) 
		self.transposed = transposed
		self.pad_mode = pad_mode
		self.c_groups = c_groups
		self.stride  = stride
		self._dilation = dilation
		self._setpad()

	@property
	def weight(self):
		return self._weight.data
	@weight.setter
	def weight(self, data):
		if self.transposed:
			data = adjoint(data)
		if data.shape != self._weight.data.shape:
			raise ValueError(f"expected shape {self._weight.data.shape} "
			                 f"but got shape {data.shape}.")
		self._weight.data = data
		self._setpad()

	@property
	def dilation(self):
		return self._dilation
	@dilation.setter
	def dilation(self, d):
		self._dilation = d
		self._setpad()

	def _setpad(self):
		ks = self.weight.shape[-1]
		p1 = int(self._dilation * np.floor((ks-1)/2))
		p2 = int(self._dilation * np.ceil((ks-1)/2))
		self._pad = (p1,p2,p1,p2)
		self._output_padding = nn.ConvTranspose2d(1,1,ks,stride=self.stride,dilation=self._dilation)._output_padding

	def _adjoint(self, x):
		C = x.shape[1] // self._weight.shape[0] if self.c_groups else 1
		# F.conv_transpose2d is funky for stride==1 ...
		# also, it can't do other padding modes than zeros.
		# Pytorch Bug: F.pad with pad = (1,0,1,0) in circular DOES NOT PAD.
		# So instead we flip the image, pad with (0,1,0,1), and flip back
		if self.stride == 1 or self.pad_mode != 'constant':
			W = torch.cat([self._weight]*C,dim=1)
			ups_x = upsample(x, self.stride, offset=0)
			pad_x = F.pad(ups_x.flip(2,3), self._pad, mode=self.pad_mode).flip(2,3)
			return F.conv2d(pad_x, adjoint(W), stride=1, dilation=self._dilation, groups=C)
		W = torch.cat([self._weight]*C,dim=0)
		output_size = (x.shape[0], x.shape[1], self.stride*x.shape[2], self.stride*x.shape[3])
		op = self._output_padding(x, output_size, 
		                          (self.stride, self.stride), 
		                          (self._pad[0], self._pad[0]), 
		                          (W.shape[3], W.shape[3]))
		return F.conv_transpose2d(x, W,
		                          padding = self._pad[0], 
		                          stride  = self.stride,
		                          output_padding = op,
		                          dilation = self._dilation,             
		                          groups = C)

	def _forward(self, x):
		C = x.shape[1] if self.c_groups else 1
		W = torch.cat([self._weight]*C)
		pad_x = F.pad(x, self._pad, mode=self.pad_mode)
		return F.conv2d(pad_x,W,stride=self.stride,dilation=self._dilation,groups=C)

class TreeConv2d(nn.Module):
	""" Tree Structured convolution.
	"""
	def __init__(self, nics, nocs, ks, levels, shared=False, transposed=False, **kwargs):
		super(TreeConv2d, self).__init__()
		if transposed:
			self.forward = self._adjoint
			self.adjoint = self._forward
		else:
			self.forward = self._forward
			self.adjoint = self._adjoint
		self.levels = levels
		nics = self.__expand__(nics)
		nocs = self.__expand__(nocs)
		ks   = self.__expand__(ks)
		argsfun = lambda i: (nics[i], nocs[i], ks[i])
		if not shared:
			self._conv = nn.ModuleList([Conv2d(*argsfun(i),transposed=transposed,**kwargs) for i in range(levels)])
		else:
			A = Conv2d(*argsfun(i),transposed=transposed,**kwargs)
			self._conv = nn.ModuleList([A for _ in range(levels)])
		self.shared = shared
		self.nics = nics
		self.nocs = nocs
		self.ks   = ks
		self.transposed = transposed
		self.pad_mode = kwargs["pad_mode"]
		self.c_groups = kwargs["c_groups"]
		self.stride   = kwargs["stride"]
		
	def __expand__(self, obj):
		if issubclass(type(obj), (tl.TensorList, list, tuple)) and len(obj) == self.levels:
			return obj
		else:
			return [obj]*self.levels
	@property
	def _weight(self):
		return tl.TensorList([c._weight for c in self._conv])
	@property
	def weight(self):
		return tl.TensorList([c.weight for c in self._conv])
	@weight.setter
	def weight(self, data):
		data = self.__expand__(data)
		for i in range(self.levels):
			self._conv[i].weight = data[i]

	def _adjoint(self, x, ret_all=False):
		if ret_all:
			lp_list = tl.TensorList([None]*len(x))
			lp_list[-1] = x[-1][:,0:1]
		t = self._conv[-1]._adjoint(x[-1])
		for i in range(self.levels-1, 0, -1):
			if ret_all:
				lp_list[i-1] = t
			t = torch.cat([t, x[i-1]], dim=1)
			t = self._conv[i-1]._adjoint(t)
		if ret_all:
			return t, lp_list
		return t

	def _forward(self, x):
		xout = tl.TensorList([None]*self.levels)
		for i in range(0, self.levels-1):
			x = self._conv[i]._forward(x[:,:self.nics[i]])
			xout[i] = x[:,self.nics[i+1]:]
		xout[-1] = self._conv[-1]._forward(x[:,:self.nics[-1]])
		return xout

