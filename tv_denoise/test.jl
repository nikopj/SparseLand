#!/usr/bin/env julia
using Plots, ProgressMeter

plt = plot([sin, cos], zeros(0), leg=false, xlims=(0,2π), ylims=(-1,1))
anim = Animation()
@showprogress for x = range(0, stop=2π, length=20)
	push!(plt, x, Float64[sin(x), cos(x)])
	frame(anim)
end 

gif(anim, "test.gif", fps=15)

