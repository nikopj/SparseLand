#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn.functional as F
import tensorlist as tl
import wvlt

def powerMethod(A, b, max_iter=1000, tol=1e-4, verbose=False):
	""" Power method for finding largest eigenvalue (and vector) 
	of linear operator
	"""
	eig_hist = np.empty(max_iter)
	for i in range(max_iter):
		b = A(b)
		b = b / torch.norm(b)
		eig_hist[i] = torch.sum(b*A(b)).item() # rayleigh quotient b^TAb
		# convergence condition
		if verbose:
			print(f"i={i:>03d}| L={eig_hist[i]:<.3e}")
		if i>1 and np.abs(eig_hist[i] - eig_hist[i-1]) < tol:
			break
	return eig_hist[i], b, eig_hist[:i+1]

def FITA(A, AT, y, z0=None, mu=1, L=None, 
		 max_iter=100,
		 tol=1e-4, 
		 verbose=False, 
		 fcn=None,
		 threshfun=wvlt.ST,
		 obj2=None,
		 hc_iters=1,
		 hc_gamma=1):
	""" Fast Iterative Soft Thresholding Algorithm
	solves: x* = (1/2)||Ax-y||^2_2 + mu*||x||_1
	A: synthesis transform
	AT: analysis transform
	y: signal
	x0: optional warm-start
	mu: sparsity penalty
	L: Lipshitz constant of A^TA
	"""
	if L is None:
		L, _, _ = powerMethod(lambda x: AT(A(x)), torch.randn_like(AT(y)), 
							  max_iter=max_iter, 
							  tol=tol, 
							  verbose=verbose)
	hist = {"obj1": np.empty(max_iter),
			"obj2": np.empty(max_iter),
			"total":np.empty(max_iter),
			"diff": np.empty(max_iter)}
	obj1 = lambda x: 0.5*torch.flatten((A(x)-y)**2).sum().item()
	if obj2 is None:
		obj2 = lambda x, mu: torch.flatten(torch.norm(x*mu,1)).sum().item()
	# initialization
	if z0 is None:
		z0 = torch.zeros_like(AT(y))
	x_old = z0.clone()
	t_old = 1
	z = z0.clone()
	# FISTA loop
	for i in range(max_iter):
		# compute iterates
		x_new = threshfun(z - AT(A(z)-y)/L, mu/L)
		t_new = (1+np.sqrt(1+4*t_old**2))/2
		xdiff = x_new-x_old
		z = x_new + xdiff*(t_old-1)/t_new
		# set vars
		x_old = x_new
		t_old = t_new
		hist["diff"][i]  = torch.flatten(torch.abs(xdiff)).max().item()
		hist["obj1"][i]  = obj1(x_new); hist["obj2"][i] = obj2(x_new, mu)
		hist["total"][i] = hist["obj1"][i] + hist["obj2"][i]
		if verbose: # print objective values
			print(f"i={i:>03d}| OT={hist['total'][i]:1.2e} | " \
				+ f"O1={hist['obj1'][i]:1.2e} | " \
				+ f"O2={hist['obj2'][i]:1.2e} | xd={hist['diff'][i]:1.2e}")
		# convergence condition
		if hist["diff"][i] < tol:
			break
		# optional function input
		if fcn is not None:
			fcn(x_new, i)
	for key in hist:
		hist[key] = hist[key][:i]
	return x_new, hist

def ITCA(A, AT, y, z0=None, mu=1, gamma=0.8, K=5,
		 max_iter=100,
		 tol=1e-4, 
		 verbose=False, 
		 fcn=None,
		 threshfun=wvlt.ST,
		 obj2=None):
	""" Iterative Soft Thresholding with Homotopy Continuation Algorithm
	solves: x* = (1/2)||Ax-y||^2_2 + mu*||x||_1
	A: synthesis transform
	AT: analysis transform
	y: signal
	x0: optional warm-start
	mu: sparsity penalty
	L: Lipshitz constant of A^TA
	"""
	hist = {"obj1": np.empty(max_iter),
			"obj2": np.empty(max_iter),
			"total":np.empty(max_iter),
			"diff": np.empty(max_iter)}
	obj1 = lambda x: 0.5*torch.flatten((A(x)-y)**2).sum().item()
	if obj2 is None:
		obj2 = lambda x, mu: torch.flatten(torch.norm(x*mu,1)).sum().item()
	# initialization
	if z0 is None:
		z0 = torch.zeros_like(AT(y))
	z = z0.clone()
	# ITSC loop
	for i in range(max_iter):
		z_old = z
		for k in range(K):
			z = threshfun(z - AT(A(z)-y), mu)
		mu = mu*gamma
		hist["diff"][i]  = torch.flatten(torch.abs(z-z_old)).max().item()
		hist["obj1"][i]  = obj1(z); hist["obj2"][i] = obj2(z, mu)
		hist["total"][i] = hist["obj1"][i] + hist["obj2"][i]
		if verbose: # print objective values
			print(f"i={i:>03d}| OT={hist['total'][i]:1.2e} | " \
				+ f"O1={hist['obj1'][i]:1.2e} | " \
				+ f"O2={hist['obj2'][i]:1.2e} | xd={hist['diff'][i]:1.2e}")
		# convergence condition
		if hist["diff"][i] < tol:
			break
		# optional function input
		if fcn is not None:
			fcn(z, i)
	for key in hist:
		hist[key] = hist[key][:i]
	return z, hist
