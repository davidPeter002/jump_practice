using JuMP
using Noise
using Plots
using Ipopt
using Statistics

function predict3d(state0, Δt)
    a = state0[3]
    v = state0[2] + a * Δt
    x = state0[1] + (state0[2] * Δt + 0.5 * a * Δt^2)
    return [x v a]
end

state₀ = zeros(30, 3)
state₀[1,:] = [10 0 0]
for i in 1:10
    state₀[i+1,:] = predict3d(state₀[i,:], 1)
end
state₀[11,3] = 1
for i in 11:20
    state₀[i+1,:] = predict3d(state₀[i,:], 1)
end
state₀[21,3] = 0
for i in 21:29
    state₀[i+1,:] = predict3d(state₀[i,:], 1)
end

truth = state₀[:,1]
δx = 2 
measure = truth
measure = add_gauss(truth, δx)

## dummy model -- trust measurement
N = 30
dummy = Model(Ipopt.Optimizer)
@variable(dummy, x[1:N])
@NLobjective(dummy, Min, sum((measure[i,1] - x[i])^2 for i = 1:N))
optimize!(dummy)
result_x = broadcast(value, x)

## free model -- balance predict and measurement
N = 30
free = Model(Ipopt.Optimizer)
@variables(free, begin
    x[1:N]
    v[1:N]
    a[1:N]
end)

@NLexpression(free,
    dm[j=1:N], (measure[j] - x[j])^2
)
@NLexpression(free,
    dp[j=1:N], 
    sum((x[j] - (x[i] + v[i]*(j-i) + 0.5*a[i]*(j-i)^2*sign(j-i)))^2 for i = max(j-2, 1):min(j+2,N)) / 4
)

@NLobjective(free, Min, sum(dm[i] + 9*dp[i] for i in 1:N))
optimize!(free)
result_x = broadcast(value, x)

## plot results
plot(1:N, truth)
scatter!(1:N, measure)
plot!(1:N, result_x)