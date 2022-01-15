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
δx = 2 
δy = 1
measure = truth[:,1:2]
measure[:,1] = add_gauss(truth[:,1], δx)
measure[:,2] = add_gauss(truth[:,2], δy)


## Generate model
N = 30
W = 1
free = Model(Ipopt.Optimizer)

# Decision Variables
@variables(free, begin
    x[1:N] 
    y[1:N] 
    v[1:N] ≥ 0
    θ[1:N]
    ω[1:N]
    a[1:N]
end)

measure_loss = 
@NLexpression(free,
    [i=1:N], (x[i] - measure[i,1])^2 + (y[i] - measure[i,2])^2
)

predict_loss = 
@NLexpression(free,
    [j=1:N],  sum(
        (x[j] - (x[i] + (v[i]*(j-i) + 0.5*a[i]*(j-i)^2*sign(j-i))*cos(θ[i])))^2 + 
        (y[j] - (y[i] + (v[i]*(j-i) + 0.5*a[i]*(j-i)^2*sign(j-i))*sin(θ[i])))^2
        for i = max(1, j-W) : min(N, j+W)) / (2W)
)

# Object Function

@NLobjective(free, Min, sum(measure_loss[i] + 9 *predict_loss[i] for i in 1:N))

optimize!(free)

result_x = broadcast(value, x)
result_y = broadcast(value, y)

plot(1:30, truth[:,1])
plot!(1:30, measure[:,1])
scatter!(1:30, measure[:,1])
plot!(1:N, result_x)