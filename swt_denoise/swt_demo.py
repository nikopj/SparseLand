#!/usr/bin/env python3
import sys, os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pywt
from pywt import threshold, coeffs_to_array, array_to_coeffs

# Stationary Wavelet Transform Denoising Demo
def main():
	# we load a small image so computations are more managable
	# params to change:
	wname = 'db5'
	levels = 4
	img_file = 'barbara.tiff'
	noise_std = 0.1

	y = imload(img_file)[64:64+128,256:256+128]
	yn = y + noise_std*np.random.randn(*y.shape)

	cropped = lambda y: y[:64,:64]
	y_small, yn_small = cropped(y), cropped(yn)

	swt, iswt = get_swt(y_small.shape, wname, levels)
	dwt, idwt = get_swt(y_small.shape, wname, levels)

	normalize = lambda x: (x-np.min(x))/np.max(x)*255
	PSNR = lambda x: -10*np.log10(x)
	MSE = lambda x,y: np.mean((x-y)**2)

	N = 10 # number of values to search
	mu = np.linspace(1e-4,0.3,N)
	swt_psnr = np.empty(N)
	dwt_psnr = np.empty(N)
	print("Testing Thresholds...")
	for i in range(N):
		print(f"mu={mu[i]:.3e}")
		# SWT
		x, hist = FISTA(iswt, swt, yn_small, mu=mu[i], L=1,
		                max_iter=20, 
						tol=1e-2,
						verbose=True, 
						fcn=None)
		yhat = iswt(x)
		swt_psnr[i] = PSNR(MSE(yhat, y_small))
		# DWT
		z = dwt(yn_small)
		zt = threshold(z, mu[i])
		yhat = idwt(zt)
		dwt_psnr[i] = PSNR(MSE(yhat, y_small))

	# threshold search figure
	fig = plt.figure()
	plt.plot(mu, swt_psnr, '-or', label='swt')
	plt.plot(mu, dwt_psnr, '-ob', label='dwt')
	plt.legend()
	plt.title('denoising threshold search')
	plt.xlabel('threshold value, mu')
	plt.ylabel('PSNR dB')
	fig.savefig('threshold_search.png')

	# find index of greatest PSNR
	swt_i0 = np.argmax(swt_psnr)
	dwt_i0 = np.argmax(dwt_psnr)

	# to save gif of sparse code generation
	gif_dir = f"GIF_denoise_{wname}_L{levels}_mu{mu[swt_i0]:.3f}"
	if not os.path.isdir(gif_dir):
		os.mkdir(gif_dir)
	save_csc_iter = lambda x, i: saveimg(normalize(np.abs(x)), os.path.join(gif_dir,f"{i:03d}.jpg"))

	# transforms for using larger image size
	swt, iswt = get_swt(y.shape, wname, levels)
	dwt, idwt = get_dwt(y.shape, wname, levels)

	print(f"Computing with best threshold, mu={mu[swt_i0]:.3f}")
	x, hist = FISTA(iswt, swt, yn, mu=mu[swt_i0], L=1,
					max_iter=100, 
					tol=1e-3,
					verbose=True, 
					fcn=save_csc_iter)
	yhat = iswt(x)
	print(f"Final SWT PSNR: {PSNR(MSE(yhat,y)):2.3f}")
	saveimg(normalize(np.abs(x)), f"swt_coeffs_denoise_{wname}_L{levels}.png")
	saveimg(normalize(np.abs(yhat)), f"swt_denoise_{wname}_L{levels}.png")

	z = dwt(yn)
	zt = threshold(z, mu[dwt_i0])
	yhat = idwt(z)
	print(f"Final DWT PSNR: {PSNR(MSE(yhat,y)):2.3f}")
	saveimg(normalize(np.abs(z)), f"dwt_coeffs_denoise_{wname}_L{levels}.png")
	saveimg(normalize(np.abs(yhat)), f"dwt_denoise_{wname}_L{levels}.png")

	PSNRCurve(idwt, z, y, path='psnr.png')

def get_swt(shape, wname, levels):
	""" returns stationary wavelet transforms for a given image shape
	"""
	# get slices for pywt array <--> coeff conversion
	coeffs = pywt.swt2(np.zeros(shape), wname, levels, trim_approx=True)
	_, swt_slices = coeffs_to_array(coeffs)
	# stationary/undecimated wavelet transform
	iswt = lambda x: pywt.iswt2(array_to_coeffs(x, swt_slices, 'wavedec2'), wname)
	swt = lambda x: coeffs_to_array(pywt.swt2(x, wname, levels, trim_approx=True))[0]	
	return swt, iswt

def get_dwt(shape, wname, levels):
	""" returns discrete wavelet transforms for a given image shape
	"""
	# get slices for pywt array <--> coeff conversion
	coeffs = pywt.wavedec2(np.zeros(shape), wname, levels)
	x0, dwt_slices = coeffs_to_array(coeffs)
	# discrete wavelet transform
	idwt = lambda x: pywt.waverec2(array_to_coeffs(x, dwt_slices, 'wavedec2'), wname)
	dwt = lambda x: coeffs_to_array(pywt.wavedec2(x, wname, levels))[0]	
	return dwt, idwt

def saveimg(arr, path):
	""" save grayscale image to path
	"""
	im = Image.fromarray(arr).convert("L")
	im.save(path)

def FISTA(A, AT, y, x0=None, mu=1, L=None, 
          max_iter=100,
		  tol=1e-4, 
		  verbose=False, 
		  fcn=None):
	""" Fast Iterative Soft Thresholding Algorithm
	solves: x* = (1/2)||Ax-y||^2_2 + mu*||x||_1
	A: synthesis transform
	AT: analysis transform
	y: signal
	x0: optional warm-start
	mu: sparsity penalty
	L: Lipshitz constant of A^TA
	"""
	if x0 is None:
		x0 = np.random.randn(*AT(y).shape)
	if L is None:
		L, _, _ = powerMethod(lambda x: AT(A(x)), x0, 
		                      max_iter=max_iter, 
							  tol=tol, 
							  verbose=verbose)
	hist = {"obj1": np.empty(max_iter),
	        "obj2": np.empty(max_iter),
			"diff": np.empty(max_iter)}
	obj1 = lambda x: 0.5*np.linalg.norm(A(x)-y)
	obj2 = lambda x: mu*np.linalg.norm(x,1)
	# initialization
	x_old = x0
	t_old = 1
	z = x0
	# FISTA loop
	for i in range(max_iter):
		# compute iterates
		x_new = threshold(z - AT(A(z)-y)/L, mu/L)
		t_new = (1+np.sqrt(1+4*t_old**2))/2
		xdiff = x_new-x_old
		z = x_new + (t_old-1)*(xdiff)/t_new
		# set vars
		x_old = x_new
		t_old = t_new
		# print objective values
		hist["diff"][i] = np.max(np.abs(xdiff)) 
		if verbose:
			hist["obj1"][i], hist["obj2"][i] = obj1(x_new), obj2(x_new)
			print(f"i={i:>03d}| O1={hist['obj1'][i]:1.2e} | " \
			    + f"O2={hist['obj2'][i]:1.2e} | xd={hist['diff'][i]:1.2e}")
		# convergence condition
		if hist["diff"][i] < tol:
			break
		# optional function input
		if fcn is not None:
			fcn(x_new, i)
	return x_new, hist

def powerMethod(A, b, max_iter=1000, tol=1e-4, verbose=False):
	""" Power method for finding largest eigenvalue (and vector) 
	of linear operator
	"""
	eig_hist = np.empty(max_iter)
	for i in range(max_iter):
		b = A(b)
		b = b / np.linalg.norm(b)
		eig_hist[i] = np.vdot(b,A(b)) # rayleigh quotient b^TAb
		# convergence condition
		if verbose:
			print(f"i={i:>03d}| L={eig_hist[i]:<.3e}")
		if i>1 and np.abs(eig_hist[i] - eig_hist[i-1]) < tol:
			break
	return eig_hist[i], b, eig_hist[:i+1]

def PSNRCurve(recon, z, y, path=None):
	""" Get M-Term approximation curve of coefficients
	recon: function to reconstruction (ie. y ~ recon(z))
	z: coefficients
	y: signal
	"""
	PSNR = lambda x: -10*np.log10(x)
	MSE = lambda x, y: np.mean(np.abs(x-y)**2.0)
	z_sort = np.sort(np.abs(z), axis=None)[::-1]
	num = z.size
	N = 100
	frac = np.linspace(0,1,N)
	psnr = np.empty(N)
	eps_min = 1e-10 # machine precision for float32s on scale of 1e-3 approx 1/255
	
	for i in range(N):
		thr = z_sort[int(np.floor((num-1)*frac[i]))]
		zt = z*(np.abs(z)>=thr)
		yhat = recon(zt)
		mse = MSE(y, yhat)
		if mse < eps_min:
			mse = eps_min
		psnr[i] = PSNR(mse)
	rel_frac = frac*num/y.size

	if path is not None:
		print('Final PSNR:', psnr[-1])
		fig = plt.figure()
		plt.tight_layout()
		plt.plot(rel_frac, psnr)
		plt.xlabel('fraction of coefficients')
		plt.ylabel('PSNR')
		fig.savefig(path)
		plt.close(fig)
	return psnr, rel_frac

def imload(path):
	""" load grayscale image from path
	"""
	im = Image.open(path).convert("L") # grayscale conversion
	return np.asarray(im)/255.0

if __name__ == "__main__":
	main()
