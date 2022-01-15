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

σx₂ = 0.5
σy₂ = 2 
measure₂ = truth[:,1:2]
measure₂[:,1] = add_gauss(truth[:,1], σx₂)
measure₂[:,2] = add_gauss(truth[:,2], σy₂)

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
    3 ≥ a[1:N] ≥ -3
end)

measure_loss₁ = 
@NLexpression(free,
    [i=1:N], (x[i] - measure₁[i,1])^2/(σx₁^2) + (y[i] - measure₁[i,2])^2/(σy₁^2)
)

measure_loss₂ = 
@NLexpression(free,
    [i=1:N], (x[i] - measure₂[i,1])^2/(σx₂^2) + (y[i] - measure₂[i,2])^2/(σy₂^2)
)

predict_loss = 
@NLexpression(free,
    [j=1:N],  sum(
        (x[j] - (x[i] + (v[i]*(j-i) + 0.5*a[i]*(j-i)^2*sign(j-i))*cos(θ[i])))^2 + 
        (y[j] - (y[i] + (v[i]*(j-i) + 0.5*a[i]*(j-i)^2*sign(j-i))*sin(θ[i])))^2
        for i = max(1, j-W) : min(N, j+W)) / (2W)
)

acceleration_loss = @NLexpression(free, [i=1:N], a[i]^2)

# Velocity Constraint
@constraint(free, [i=2:N], v[i] == v[i-1] + a[i-1])

# Rotation Constraint
@constraint(free, [k=2:N], θ[k] == θ[k-1] + ω[k-1])

# Acceleration Constraint
@constraint(free, [i=1:N], sum((a[j] - a[i])^2 for j = max(1, i-2):min(N, i+2)) / 4 <= 1)

# Object Function
@NLobjective(free, Min, sum(measure_loss₁[i] + measure_loss₂[i] + 9*predict_loss[i] + acceleration_loss[i] for i in 1:N))

optimize!(free)

result_x = broadcast(value, x)
result_y = broadcast(value, y)
result_v = broadcast(value, v)
result_θ = broadcast(value, θ)
result_θ = broadcast(rad2deg, result_θ).%360
result_ω = broadcast(value, ω)
result_a = broadcast(value, a)

l = @layout [a b ; c d ; e f]
## plot - x
p₁ = plot(1:N, truth[:,1])
plot!(1:N, measure₁[:,1])
plot!(1:N, measure₂[:,1])
plot!(1:N, result_x)

## plot - y
p₂ = plot(1:N, truth[:,2])
plot!(1:N, measure₁[:,2])
plot!(1:N, measure₂[:,2])
plot!(1:N, result_y)

## plot - v
p₃ = plot(1:N, truth[:,3])
plot!(1:N, result_v)

## plot - θ
p₄ = plot(1:N, broadcast(rad2deg,truth[:,4]))
plot!(1:N, result_θ)

## plot - ω
p₅ = plot(1:N, truth[:,5])
plot!(1:N, result_ω)

## plot - a
p₆ = plot(1:N, truth[:,6])
plot!(1:N, result_a)

plot(p₁, p₂, p₃, p₄, p₅, p₆, layout=l)