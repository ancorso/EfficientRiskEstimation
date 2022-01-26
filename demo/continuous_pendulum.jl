using POMDPs, POMDPGym, POMDPSimulators, POMDPPolicies, Distributions, Plots
using Crux, Flux, BSON, ImportanceWeightedRiskMetrics
include("utils.jl")
include("discrete_exploration.jl")

## Setup and solve the mdp
# dt = 0.1
# mdp = InvertedPendulumMDP(λcost=1, Rstep=.1, dt=dt, px=Normal(0f0, 0.1f0))
# policy = ActorCritic(GaussianPolicy(ContinuousNetwork(Chain(Dense(2, 32, relu), Dense(32, 1, tanh), x-> x .* 2f0), 1), [0f0]), 
#                      ContinuousNetwork(Chain(Dense(2, 32, relu), Dense(32, 1))))
# policy = solve(PPO(π=policy, S=state_space(mdp), N=100000, ΔN=400, max_steps=400), mdp)
# BSON.@save "policies/pendulum_policy.bson" policy

policy = BSON.load("policies/pendulum_policy.bson")[:policy]
# Crux.gif(mdp, policy, "out.gif", max_steps=100,)


## Construct the risk estimation mdp where actions are disturbances
dt = 0.1 # Do not change
maxT = 2.0 # Do not change
px_nom = Normal(0f0, 0.35f0) # Do not change
# px_nom = Normal(0f0, 1.0f0)

# px = DistributionPolicy(px_nom)
px = GaussianPolicy(ContinuousNetwork(Chain(x -> zeros(Float32, 1, size(x)[2:end]...)), 1), ContinuousNetwork(Chain(x ->  Float32(Base.log(0.35f0)) .* ones(Float32, 1, size(x)[2:end]...)), 1))

# cost environment
env = InvertedPendulumMDP(dt=dt, failure_thresh=Inf)
costfn(m, s, sp) = isterminal(m, sp) ? abs(sp[2]) : 0
rmdp = RMDP(env, policy, costfn, true, dt, maxT)

# Failure environment
# env = InvertedPendulumMDP(dt=dt, failure_thresh=π/2)
# costfn(m, s, sp) = isterminal(m, sp) ? isfailure(m, sp) : 0
# rmdp = RMDP(env, policy, costfn, true, dt, maxT)

## Get ground trutch estimates
# mc_samps = [simulate(RolloutSimulator(), rmdp, FunctionPolicy((s) -> action(px, rand(3))[1])) for _=1:Int(1e7)]
# BSON.@save "data/10mil_mcsamps_continuouspendulum.bson" mc_samps 

mc_samps = BSON.load("data/10mil_mcsamps_continuouspendulum.bson")[:mc_samps]
mc_weights = ones(length(mc_samps))

# compute risk metrics
risk_metrics = IWRiskMetrics(mc_samps, mc_weights, 0.001)
risk_metrics.cvar
risk_metrics.var
risk_metrics.mean
risk_metrics.worst

α = [0.01, 1e-5] # allowable range of α

## Setup and run deep rl approach
A() = GaussianPolicy(ContinuousNetwork(Chain(Dense(3, 64, relu), Dense(64, 1, tanh), x->x.*2f0), 1), zeros(Float32, 1))
D_CDF() = ContinuousNetwork(Chain(Dense(4, 64, relu), Dense(64, 1, sigmoid))) # Low priority, hyperparameter
D_CVaR() = ContinuousNetwork(Chain(Dense(4, 64, relu), Dense(64, 1, sigmoid), x->x.*3.15f0), 1) # Low priority, hyperparameter
AC1() = ActorCritic(A(), D_CDF())
AC2() = ActorCritic(A(), D_CVaR())

function log_err_pf(π, D, ys)
    N = length(ys)
    sum([abs.(log.(value(n, D[:s], D[:a]) .+ eps())  .-  log.(y  .+ eps())) for (n, y) in zip(π.networks[1:N], ys)])    
end

function abs_err_pf(π, D, ys)
    N = length(ys)
    sum([abs.(value(n, D[:s], D[:a])  .-  y) for (n, y) in zip(π.networks[1:N], ys)])
end
    
𝒮 = CERL_Continuous(π=MixtureNetwork([AC1(), AC2(), px], [0.45, 0.45, 0.1]), 
                  px=px,
                  priority_fn=log_err_pf, #log_err_pf, abs_err_pf
                  α=1e-3, # experiment parameter
                  prioritized=true, # false <- ablation
                  use_likelihood_weights=false, #false (hyperparameter)
                  pre_train_callback=compute_risk_cb(1000, 0.1), # [0.01, 0.1, 0.5] 0.1 is the minmum fraction of samples in tail. (hyperparameter)
                  log=(;period=1000),
                  N=100000, # Number of steps in the environment
                  c_loss=multi_td_loss(names=["Q_CDF", "Q_CVaR"], loss=Flux.Losses.msle, weight=:weight), # if prioritized, then weight=nothing. loss can be: [Flux.Losses.msle, Flux.Losses.mse] <- ablation
                  a_loss=multi_actor_loss(Crux.IS_L_KL_log, 2),
                  S=state_space(rmdp))
solve(𝒮, rmdp)



# value(𝒮.agent.π.networks[1], ones(4))
# 𝒟 = buffer_like(𝒮.buffer, capacity=𝒮.c_opt.batch_size, device=device(𝒮.agent.π))
# Crux.rand!(𝒟, 𝒮.buffer, 𝒮.extra_buffers..., fracs=𝒮.buffer_fractions, i=𝒮.i)
# Crux.train!(actor(𝒮.agent.π), (;kwargs...) -> 𝒮.a_opt.loss(𝒮.agent.π, 𝒮.𝒫, 𝒟; kwargs...), 𝒮.a_opt, info=Dict())
# value(𝒮.agent.π.networks[1], ones(4))

all(isfinite.(𝒮.buffer[:weight]))

extrema(𝒮.buffer[:weight])
extrema(𝒮.buffer.priority_params.priorities[1:length(𝒮.buffer)])



drl_samps, drl_weights = get_samples(𝒮.buffer, px)


histogram(exploration(𝒮.agent.π.networks[1], 𝒮.buffer[:s])[1][:])
histogram!(exploration(𝒮.agent.π.networks[2], 𝒮.buffer[:s])[1][:])






g = A()
using Zygote
function loss(s)
    # x = Zygote.ignore() do
    #     rand(px_nom, size(s, 2))
    # end
    x, logqx = exploration(g, s)
    # logqx = logpdf(g, s, x)
    logpx = logpdf(px, s, x) .+ log.(sigmoid.(10f0 .* x))
    # Zygote.ignore() do 
    # 
    # end

    -mean(exp.(logpx .- logqx) .* logqx)
end

px

opt = ADAM(1e-3)
parameters = Flux.params(g)

std(rand(px_nom, 1024))
histogram(rand(px_nom, 1024), label="nom")
# histogram!(rand(Normal(), 1024), alpha=0.1, label="unit")
histogram!(exploration(g, randn(Float32, 3, 1024))[1][:], alpha=0.3, label="iter 0")


for i=1:1000
    println("i=$i, mean: ", mean(g.μ(randn(Float32, 3,1024))), " std: ", exp.(g.logΣ(randn(Float32, 3,1024)))[1])
    s = rand(Float32, 3,1024)
    grads = Flux.gradient(parameters) do
        loss(s)
    end
    Flux.Optimise.update!(opt, parameters, grads)
end

histogram!(exploration(g, randn(Float32, 3, 1024))[1][:], label="iter 1000", alpha=0.3)


risk_metrics = IWRiskMetrics(drl_samps, drl_weights, 0.001)

#TODO: Compute error to ground truth abs(gt - est) / gt



## Plot the value function map
heatmap(-3:0.01:3, -8:0.01:8, (x,y)->value(𝒮.agent.π.networks[1], [1.9, x, y, 0])[1], xlabel="θ", ylabel="ω") # cdf
indices = 𝒮.buffer[:s][1,:] .≈ 1.9
scatter!(𝒮.buffer[:s][2,indices], 𝒮.buffer[:s][3,indices], color = Int.(abs.(𝒮.buffer[:s][2,indices]) .> 𝒮.𝒫.rα[1]))

 
heatmap(-3:0.1:3, -8:0.1:8, (x,y)->value(𝒮.agent.π.networks[2], [1.9, x, y, 1])[1], xlabel="θ", ylabel="ω")
scatter!(𝒮.buffer[:s][2,indices], 𝒮.buffer[:s][3,indices], color = Int.(abs.(𝒮.buffer[:s][2,indices]) .> 𝒮.𝒫.rα[1]))

indices = 𝒮.buffer[:s][1,:] .≈ 1.9
scatter!(𝒮.buffer[:s][2,indices], 𝒮.buffer[:s][3,indices], color = Int.(abs.(𝒮.buffer[:s][2,indices]) .> 𝒮.𝒫.rα[1]))

extrema(𝒮.buffer[:s][3,indices])

𝒮.buffer[:likelihoodweight]
plot(log.(𝒮.buffer[:likelihoodweight][:]))


means = 𝒮.agent.π.networks[1].A.μ(𝒮.buffer[:s])
vars = exp.(𝒮.agent.π.networks[1].A.logΣ(𝒮.buffer[:s]))
plot(means[:][1:1000])

histogram(𝒮.buffer[:a][:])


## Plot the distribution of costs
histogram(drl_samps, bins=0:0.1:3)
histogram!(mc_samps, bins=0:0.1:3)

# Make the plots
make_plots([mc_samps, drl_samps], [mc_weights, drl_weights], ["MC", "DRL"], 1e-5)

