import pywt
import numpy as np
import torch
import torch.nn.functional as F
import tensorlist as tl
import utils, conv

def main():
	""" Verify that conv.Conv2d and conv.TreeConv2d are implemented 
	correctly by performing a discrete wavelet transform with them.
	DWT, IDWT = getop_wavelet('db3',J=5,stationary=False)
	x = torch.cat([utils.imgLoad(f"Set12/{i:02d}.png") for i in [1]])
	z = DWT(x)
	print("z.shape =", z.shape)
	z = wthresh(z, 0)
	for zz in z:
		fig = utils.visplot(zz.transpose(0,1).abs()); fig.show()
	xhat = IDWT(z)
	err = torch.mean((x-xhat)**2)
	print(f"MSE = {err:.2e}")
	fig = utils.visplot(xhat); fig.show()
	input()
	"""
	Wa, Ws = filter_bank_2D('db3'); ks = Wa.shape[-1]
	W1 = torch.zeros(4,1,7,7)
	W1[:,:,:-1,:-1] = Wa
	W2 = torch.zeros(4,1,7,7)
	W2[:,:,1:,1:] = Wa
	p1 = int(np.floor((ks)/2))
	p2 = int(np.ceil((ks)/2))
	DWT  = torch.nn.Conv2d(1,4,ks,stride=2, padding=p1, bias=False)
	DWT.weight.data = W1
	IDWT = torch.nn.ConvTranspose2d(4,1,ks,stride=2, padding=p1, output_padding=1, bias=False)
	IDWT.weight.data = W1
	x = torch.cat([utils.imgLoad(f"Set12/{i:02d}.png") for i in [1]])
	z = DWT(x)
	print("z.shape =", z.shape)
	xhat = IDWT(z)
	err = torch.mean(((x-xhat)[:,:,20:-20,20:-20])**2)
	print(f"MSE = {err:.2e}")
	fig = utils.visplot(xhat[:,:,10:-10,10:-10]); fig.show()
	fig = utils.visplot(z.transpose(0,1)); fig.show()
	input()

def wthresh(x,t):
	""" Wavelet thresholding. Thresholds wavelet band-pass coefficients.
	x: (TensorList) wavelet coefficients from a tree-convolution
	t: threshold value.
	"""
	zL = x[-1][:,:1]
	x[-1] = x[-1][:,1:]
	z = ST(x,t)
	z[-1] = torch.cat([zL, z[-1]],dim=1)
	return z

def ST(x,t):
	""" Soft-thresholding operation. 
	Input x, threshold t.
	"""
	return x.sign()*F.relu(x.abs()-t)

def getop_wavelet(wname, J=1, stationary=False):
	""" Return discrete wavelet transform (and inverse) 
	with J levels of decomposition (on the lowpass signal).
	"""
	if stationary:
		stride = 1; dilation = 2
	else:
		stride = 2; dilation = 1
	kwargs = {"stride": stride, "dilation": dilation, "pad_mode": "circular", "c_groups": True}
	Wa, Ws = filter_bank_2D(wname)
	noc, nic = Wa.shape[:2]
	ks = Wa.shape[-1]
	if J==1:
		DWT  = conv.Conv2d(nic,noc,ks,**kwargs)
		IDWT = conv.Conv2d(noc,nic,ks,**kwargs, transposed=True)
	else:
		DWT  = conv.TreeConv2d(nic,noc,ks,J,**kwargs)
		IDWT = conv.TreeConv2d(noc,nic,ks,J,**kwargs,transposed=True)
	DWT.weight  = Wa * DWT.stride  / np.sqrt(noc)
	IDWT.weight = Ws * IDWT.stride / np.sqrt(noc)
	DWT._weight.requires_grad = False
	IDWT._weight.requires_grad = False
	return DWT, IDWT

def filter_bank_1D(wname):
	""" Returns 1D wavelet filterbank.
	wname: wavelet name (from pyWavelets)
	"""
	# returns analysis and synthesis filters concat-ed
	fb = torch.tensor(pywt.Wavelet(wname).filter_bank).float()
	wa, ws = fb[:2,:], fb[2:,:]
	return wa, ws

def filter_bank_2D(wname):
	""" Returns 2D wavelet filterbank.
	Formed as outerproduct using return from wvlt1Dfb
	wname: wavelet name (from pywt)
	Wa: analysis fb, 1 to n channels
	Ws: synthesis fb. n to 1 channels
	"""
	wa, ws = filter_bank_1D(wname)
	Wa, Ws = nonsep(wa), nonsep(ws)
	return Wa.transpose(0,1), Ws

def outerprod(u,v):
	""" Outer-product between vectors u, v
	"""
	W = torch.einsum('...i,...j->...ij',u,v)
	return W

def nonsep(w):
	""" to non-seperable fb
	Turns 1D filter bank into 2D non-seperable filter bank.
	W: n to 1 channels 2D filter bank
	"""
	w1 = torch.cat([w[:1], w[:1], w[1:], w[1:]])
	w2 = torch.cat([w, w])
	# add dim for torch kernel, flip so corr -> conv
	W  = outerprod(w1,w2)[None,:].flip(2,3)
	return W

if __name__ == "__main__":
	main()


