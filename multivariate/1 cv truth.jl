using JuMP
using Noise
using Plots
using Ipopt
using Statistics

## CV truth track, CT model, [x,y] measure
N = 50

function predict(state0, Δt)
   ω = state0[5] 
   a = state0[6]

   θ = state0[4] + ω * Δt
   v = state0[3] + a * Δt

   x = state0[1] + (state0[3] * Δt + 0.5 * a * Δt^2) * cos(θ)
   y = state0[2] + (state0[3] * Δt + 0.5 * a * Δt^2) * sin(θ)
   return [x y v θ ω a]
end

## Generate Truth
state₀ = [-50 -50 5 deg2rad(45) 0 0] # x,y,v,θ,ω,a
truth = state₀
time = Vector(LinRange(0, 5, N))
for t in time[2:end];
    truth = [truth; predict(state₀, t)]
end

## Generate measure
δx = 0.5
δy = 1
measure = truth[:,1:2]
measure[:,1] = add_gauss(truth[:,1], δx)
measure[:,2] = add_gauss(truth[:,2], δy)

## Generate model
track = Model(Ipopt.Optimizer)

# Decision Variables
@variables(track, begin
    Δt ≥ 0, (start = 1 / N) # time step
    x[1:N] 
    y[1:N] 
    v[1:N] ≥ 0
    θ[1:N]
    ω[1:N]
    a[1:N]
end)

set_start_value.(x, 0)
set_start_value.(y, 0)
set_start_value.(v, 0)
set_start_value.(θ, 0)
set_start_value.(ω, 0)
set_start_value.(a, 0)

# Initial conditions
fix(ω[1], 0, force=true)
fix(a[1], 0, force=true)

# Dynamics
for t in 2:N
    @NLconstraint(track, ω[t] == ω[t-1])
    @NLconstraint(track, a[t] == a[t-1])
    @NLconstraint(track, v[t] == v[t-1] + (a[t] + a[t-1]) * 0.5 * Δt)
    @NLconstraint(track, θ[t] == θ[t-1] + (ω[t] + ω[t-1]) * 0.5 * Δt)
    @NLconstraint(track, x[t] == x[t-1] + (v[t] * Δt + a[t] * 0.5 * Δt^2) * cos(θ[t]))
    @NLconstraint(track, y[t] == y[t-1] + (v[t] * Δt + a[t] * 0.5 * Δt^2) * sin(θ[t]))
end

# Object Function
@NLexpression(
    track,
    residual, sum((x[i] - measure[i,1])^2 + (y[i] - measure[i,2])^2 for i = 1:N)
)

@NLobjective(track, Min, residual)

optimize!(track)

result_x = broadcast(value, x)
result_y = broadcast(value, y)

#plot(1:N, truth[:,1])
#scatter!(1:N, measure[:,1])
#plot!(1:N, result_x)
#
#plot(1:N, truth[:,2])
#scatter!(1:N, measure[:,2])
#plot!(1:N, result_y)