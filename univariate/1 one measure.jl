using JuMP
using Noise
using Plots
using Ipopt
using Statistics

N = 50
K = 20
B = -50
δ = 4
param = [K;B]

time = [Vector(LinRange(0, 5, N)) ;; ones(N)]
truth = time * param
meas = add_gauss(truth, δ)

model = Model(Ipopt.Optimizer)
@variable(model, x[1:2])
@objective(model, Min, sum((time*x - meas).^2))

optimize!(model)
optval = [value(x[1]);value(x[2])]
guess = time * optval
error = mean(broadcast(abs, guess - truth))
print(error)

plot(time[:,1], truth)
scatter!(time[:,1], meas)
plot!(time[:,1], guess)