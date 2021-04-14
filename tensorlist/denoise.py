import sys, os
import numpy as np
import torch
import matplotlib.pyplot as plt

import utils, conv, wvlt, solvers
import tensorlist as tl

def main():
	args = ('Set12/01.png', np.linspace(1e-3,2e-1,20),
	        'db3', 3, 25)
	ex_ITA_denoise(*args)

def ex_ITCA_denoise(fn, thresh_pts, wname, levels, noise_std):
	x = utils.imgLoad(fn)[:,:,128:128+64,128:128+64]
	# get stationary wavelet transform
	_, ISWT = wvlt.getop_wavelet(wname,J=levels,stationary=True)
	# remove low-pass from penalty objective function (for printing purposes only)
	obj2 = lambda x, mu: mu*(torch.flatten(torch.norm(x[:-1],1)).sum() \
	                       + torch.flatten(torch.norm(x[-1][:,1:],1)).sum())
	y = utils.awgn(x, noise_std)
	mu0 = torch.flatten(ISWT.adjoint(y)).max().item()
	print(f"mu0 = {mu0:.2e}")
	what, hist = solvers.ITCA(ISWT, ISWT.adjoint, y, mu=mu0,
	                          gamma=0.8, K=50, max_iter=10,verbose=True,
	                          threshfun=wvlt.wthresh,obj2=obj2)
	xhat = ISWT(what)
	PSNR = lambda v: -10*np.log10(torch.mean((x-v)**2).item())
	utils.visplot(torch.cat([y, xhat, x]), 
	              titles=[f"Noisy, {PSNR(y):.2f} dB", 
	                      f"Denoised, {PSNR(xhat):.2f} dB",
	                       "Ground Truth"])
	plt.figure()
	plt.semilogy(hist['total'], '-b')
	plt.xlabel("iterations")
	plt.ylabel("BPDN functional")
	plt.show()

def ex_ITA_denoise(fn, thresh_pts, wname, levels, noise_std):
	x = utils.imgLoad(fn)
	# get stationary wavelet transform
	_, ISWT = wvlt.getop_wavelet(wname,J=levels,stationary=False)
	# remove low-pass from penalty objective function (for printing purposes only)
	obj2 = lambda x, mu: mu*(torch.flatten(torch.norm(x[:-1],1)).sum() \
	                       + torch.flatten(torch.norm(x[-1][:,1:],1)).sum())
	# precompute Lipschitz-constant 
	L = solvers.powerMethod(lambda v: ISWT.adjoint(ISWT(v)), torch.randn_like(ISWT.adjoint(x[:,:,:128,:128])), max_iter=100, verbose=True)[0]
	denoise = lambda y, t: ISWT(solvers.FITA(ISWT, ISWT.adjoint, y, L=L,
	                                         mu=t, max_iter=100, verbose=True, 
	                                         threshfun=wvlt.wthresh, obj2=obj2)[0])
	# find optimal threshold on image patch
	thresh = find_threshold(x[:,:,128:128+128,128:128+128], denoise, noise_std, thresh_pts, show=False)
	# denoise with optimal threshold on full image
	y = utils.awgn(x, noise_std)
	what, hist = solvers.FITA(ISWT,ISWT.adjoint,y,L=L,mu=thresh,max_iter=500,verbose=True,threshfun=wvlt.wthresh,obj2=obj2)
	xhat = ISWT(what)
	PSNR = lambda v: -10*np.log10(torch.mean((x-v)**2).item())
	utils.visplot(torch.cat([y, xhat, x]), 
	              titles=[f"Noisy, {PSNR(y):.2f} dB", 
	                      f"Denoised, {PSNR(xhat):.2f} dB",
	                       "Ground Truth"])
	plt.figure()
	plt.semilogy(hist['total'], '-b')
	plt.xlabel("iterations")
	plt.ylabel("BPDN functional")
	plt.show()

def ex_wthresh_denoise(fn, thresh_pts, wname, levels, noise_std):
	x = utils.imgLoad(fn)
	DWT, IDWT = wvlt.getop_wavelet(wname,J=levels,stationary=False)
	denoise = lambda y,t: IDWT(wvlt.wthresh(DWT(y),t))
	thresh = find_threshold(x, denoise, noise_std, thresh_pts, show=True)

def find_threshold(x, denoise, noise_std=25, thresh_pts=np.linspace(1e-3,5e-1,5), verbose=True, show=False):
	""" Loop over thresholds to find one that gets maximum PSNR.
	x: clean input image
	denoise(y,t): function that denoises image y with value t.
	              t is a threshold value in ex_wthresh, 
	              t is a penalty value in ex_ITA.
	"""
	y = utils.awgn(x, noise_std)
	tau = thresh_pts
	best_psnr = -np.inf
	PSNR = lambda v: -10*np.log10(torch.mean((x-v)**2).item())
	psnr_list = []
	for i in range(len(tau)):
		if verbose:
			print(f"tau[{i}] = {tau[i]:.2e}")
		xhat = denoise(y, tau[i])
		psnr = PSNR(xhat)
		psnr_list.append(psnr)
		if psnr > best_psnr:
			best_psnr = psnr
			best_ind  = i
	if verbose:
		print(f"best index = {best_ind}")
		print(f"best tau = {tau[best_ind]:.2e}")
		print(f"best PSNR = {psnr_list[best_ind]:.2f}")
	if show:
		xhat = denoise(y, tau[best_ind])
		utils.visplot(torch.cat([y, xhat, x]), 
		              titles=[f"Noisy, {PSNR(y):.2f} dB", 
		                      f"Denoised, {PSNR(xhat):.2f} dB",
		                       "Ground Truth"])
		plt.figure()
		plt.plot(tau, psnr_list, '-ob', mec='k')
		plt.xlabel("threshold")
		plt.ylabel("PSNR (dB)")
		plt.show()
	return tau[best_ind]

if __name__=="__main__":
	main()

