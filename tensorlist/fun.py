#!/usr/bin/env python3
import sys
import numpy as np
import torch
import torch.nn.functional as F
import tensorlist as tl
import utils, wvlt
import matplotlib.pyplot as plt

def main():
	DWT, IDWT = wvlt.getop_wavelet('db3',J=3,stationary=False)
	img = utils.imgLoad("/home/nikopj/doc/pic/film/7480/74800029.JPG")
	gry = img.mean(dim=1,keepdim=True)
	print("img.shape =", img.shape)
	print("gry.shape =", gry.shape)
	z = DWT(gry)
	gry = z[-1][:,0:1]
	print("gry.shape =", gry.shape)
	m,n = gry.shape[2:]
	DWT, IDWT = wvlt.getop_wavelet('db3',J=4,stationary=True)
	for ii in range(300):
		print("ii =", ii)
		z = DWT(gry)
		dx, dy = z[-1][0,1].abs().numpy(), z[-1][0,2].abs().numpy()
		bp = shortest_path(dx+dy)
		fltidx = bp + n*np.arange(m)
		gry = np.delete(gry, fltidx).reshape(1,1,m,n-1)
		n -= 1
	plt.imshow(gry[0,0])
	plt.show()


def shortest_path(I):
	""" Given cost image I (m,n), return the vertical path 
	of least cost. 
	"""
	m, n = I.shape
	infpad = lambda arr: np.pad(arr, (1,1), constant_values=1e5)
	cost = infpad(I[-1,:])
	path = -1*np.ones((m-1,n));
	for i in range(m-1,0,-1):
		costp   = I[i-1,:] + np.stack((cost[0:n], cost[1:n+1], cost[2:n+2]))
		relpath = np.argmin(costp, axis=0) - 1
		cost    = infpad(np.amin(costp, axis=0))
		path[i-1,:] = np.arange(n) + relpath
	best_path = np.empty((m,)).astype(int); 
	best_path[0] = int(np.argmin(cost[1:n+1]))
	for i in range(m-1):
		best_path[i+1] = int(path[i,best_path[i]])
	return best_path

if __name__ == "__main__":
	main()
