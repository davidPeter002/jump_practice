using JuMP
using Noise
using Plots
using Ipopt
using Statistics

function predict(state0, Δt)
    ω = state0[5] 
    a = state0[6]
 
    θ = state0[4] + ω * Δt
    v = state0[3] + a * Δt
 
    x = state0[1] + (state0[3] * Δt + 0.5 * a * Δt^2) * cos(θ)
    y = state0[2] + (state0[3] * Δt + 0.5 * a * Δt^2) * sin(θ)
    return [x y v θ ω a]
end

state₀ = zeros(30, 6)
state₀[1,:] = [10 0 0 0 0 0]
for i in 1:10
    state₀[i+1,:] = predict(state₀[i,:], 1)
end
state₀[11,6] = 1
for i in 11:20
    state₀[i+1,:] = predict(state₀[i,:], 1)
end
state₀[21,6] = 0
for i in 21:29
    state₀[i+1,:] = predict(state₀[i,:], 1)
end
truth = state₀

## Generate measure
σx₁ = 2 
σy₁ = 0.5
measure₁ = truth[:,1:2]
measure₁[:,1] = add_gauss(truth[:,1], σx₁)
measure₁[:,2] = add_gauss(truth[:,2], σy₁)