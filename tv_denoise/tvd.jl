#!/usr/bin/env julia
using Plots, Printf
using SparseArrays, LinearAlgebra
using Images, TestImages, ImageView
plotly()

N = 256; M = 50; L = 20;
v = zeros(N);

a = clamp.(cumsum(rand(10:50, L)), 1, 256);

for i=1:L-1
	v[a[i]:a[i+1]] .= 2*rand(-3:3);
end

σₙ = 1;
y = v + σₙ*randn(N);

# forward Euler derivative matrix
FEDmat(N) = spdiagm(0=>-1*ones(N), 1=>ones(N))[1:N-1,1:N]; 

function FEDmat2D(M::Int, N::Int)
	# vertical derivative
	S = spdiagm(N-1, N, ones(N-1));
	T = FEDmat(M);
	Dy = kron(S,T);
	# horizontal derivative
	S = FEDmat(N);
	T = spdiagm(M-1, M, ones(M-1));
	Dx = kron(S,T);
	return [Dx; Dy];
end

HT(x,τ) = x*(abs(x) > τ);           # hard-thresholding
ST(x,τ) = sign(x)*max(abs(x)-τ, 0); # soft-thresholding

function tvd_majmin(y::AbstractVector, λ::Float64;
                   maxit      = 100,
                   tol        = 1e-6,
                   verbose    = true)
	D = FEDmat(length(y));
	return tvd_majmin(y, D, λ, maxit, tol, verbose);
end

function tvd_majmin(y::AbstractMatrix, λ::Float64;
                  maxit      = 100,
                  tol        = 1e-6,
                  verbose    = true)
	M, N = size(y);
	D = FEDmat2D(M,N);
	x, hist = tvd_majmin(vec(y), D, λ, maxit, tol, verbose);
	return reshape(x, M, N), hist
end

function tvd_majmin(y::AbstractVector, D::AbstractMatrix, λ::Float64, maxit::Int, tol::Float64, verbose::Bool)
	M, N = size(D);
	DDᵀ = D*D';
	x = y;
	Dy = D*y;
	Dx = D*x;

	objfun(x,Dx) = 0.5*sum((x-y).^2) + λ*norm(Dx, 1);
	z = zeros(M);
	F = zeros(maxit); # objective fun

	k = 0
	while k < maxit
		H = spdiagm(abs.(Dx)/λ) + DDᵀ;
		x = y - D'*(H\Dy);
		Dx = D*x;
		F[k+1] = objfun(x,Dx);
		k += 1;
		if verbose
			@printf "k: %3d | F= %.3e \n" k F[k];
		end
	end
	return x, (k=k, obj=F[1:k])
end

function tvd_admm(y::AbstractVector, λ::Float64, ρ::Float64=1;
                  maxit      = 100,
                  tol        = 1e-6,
                  verbose    = true)
	D = FEDmat(length(y));
	return tvd_admm(y, D, λ, ρ, maxit, tol, verbose)
end

function tvd_admm(y::AbstractMatrix, λ::Float64, ρ::Float64=1;
                  maxit      = 100,
                  tol        = 1e-6,
                  verbose    = true)
	M, N = size(y);
	D = FEDmat2D(M,N);
	x, hist = tvd_admm(vec(y), D, λ, ρ, maxit, tol, verbose);
	return reshape(x, M, N), hist
end

function tvd_admm(y::AbstractVector, D::AbstractMatrix, λ::Float64, ρ::Float64, maxit::Int, tol::Float64, verbose::Bool)
	M, N = size(D);
	objfun(x,Dx) = 0.5*sum((x-y).^2) + λ*norm(Dx, 1);
	x = y; 
	z = zeros(M);
	u = zeros(M);
	F = zeros(maxit); # objective fun, 
	r = zeros(maxit); # primal residual
	s = zeros(maxit); # dual residual

	C = cholesky(I + ρ*D'*D);

	k = 0;
	while k == 0 || k < maxit && r[k] > tol 
		x = C\(y + ρ*D'*(z - u));
		Dx = D*x; 
		zᵏ = z;
		z = ST.(Dx + u, λ/ρ);
		u = u + Dx - z;       # dual ascent
		r[k+1] = norm(Dx - z);
		s[k+1] = ρ*norm(D'*(z - zᵏ));
		F[k+1] = objfun(x,Dx);
		k += 1;
		if verbose
			@printf "k: %3d | F= %.3e | r= %.3e | s= %.3e \n" k F[k] r[k] s[k] ;
		end
	end
	return x, (k=k, obj=F[1:k], pres=r[1:k], dres=s[1:k])
end

λ = 0.1;
ρ = 5.0;
maxit = 500;
tol = 1e-3;
verbose = true;

σₙ = 0.1;
img = testimage("cameraman")
y = img + σₙ*randn(size(img))
x, hist = tvd_admm(y, λ, ρ; maxit=maxit, tol=tol, verbose=verbose);
 
@printf "||x - img||F = %.3e\n" norm(x-img);

imshow(x)

if false
	x1, hist1 = tvd_admm(y, λ, ρ; maxit=maxit, tol=tol, verbose=verbose);
	x2, hist2 = tvd_majmin(y, λ;  maxit=maxit, tol=tol, verbose=verbose);

	@printf "||x₁ - v||₂ = %.3e\n" norm(x1-v);
	@printf "||x₂ - v||₂ = %.3e\n" norm(x2-v);

	pv = plot(1:N, v, label="v")
	px1 = plot(1:N, x1, label="ADMM")
	px2 = plot(1:N, x2, label="MajMin")
	py = plot(1:N, y, label="y")
	plt = plot(py, px1, px2, pv, layout=(4,1))
	display(plt)

	plt = plot(1:hist1.k, hist1.obj)
	plot!(1:hist2.k, hist2.obj)
	title!("Objective value")
	xlabel!("iterations (k)")
	# pr = plot(1:k, hist.pres, label="primal")
	# plot!(1:k, hist.dres, label="dual")
	# xlabel!("iterations (k)")
	# plt = plot(pF, pr, layout=(1,2))
	display(plt)
end


