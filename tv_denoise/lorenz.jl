#!/usr/bin/env julia
using Plots, ProgressMeter
plotlyjs()

Base.@kwdef mutable struct Lorenz
	dt::Float64 = 0.02
	σ ::Float64 = 10
	ρ ::Float64 = 28
	β ::Float64 = 8/3
	x ::Float64 = 1
	y ::Float64 = 1
	z ::Float64 = 1
end

function step!(ℓ::Lorenz)
	dx = ℓ.σ * (ℓ.y - ℓ.x);       ℓ.x += ℓ.dt * dx
	dy = ℓ.x * (ℓ.ρ - ℓ.z) - ℓ.y; ℓ.y += ℓ.dt * dy
	dz = ℓ.x * ℓ.y - ℓ.β * ℓ.z  ; ℓ.z += ℓ.dt * dz
end

attractor = Lorenz()

plt = plot3d(
	1,
	xlim = (-30,30),
	ylim = (-30,30),
	zlim = (0,60),
	title= "Lorenz Attractor",
	marker=2
)

#anim = Animation()
# @showprogress for i=1:20
#
@gif for i=1:1500
	step!(attractor)
	push!(plt, attractor.x, attractor.y, attractor.z)
	#frame(anim)
end every 10

#gif(anim, "test.gif", fps=15)

