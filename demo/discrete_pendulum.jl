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
dt = 0.1
maxT = 2.0
px_nom = Normal(0f0, 0.35f0)
# px_nom = Normal(0f0, 1.0f0)

xs = [-1, 0, 1]
pxs = pdf.(px_nom, xs)
pxs = pxs ./ sum(pxs)
px = DistributionPolicy(DiscreteNonParametric(xs, pxs))

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
# BSON.@save "data10mil_mcsamps.bson" mc_samps 

mc_samps = BSON.load("data/10mil_mcsamps.bson")[:mc_samps]
mc_weights = ones(length(mc_samps))


## Setup and run deep rl approach
πexplore = DiscreteExploration((s) -> pxs)
D_CDF() = DiscreteNetwork(Chain(Dense(3, 128, relu), Dense(128, 3, sigmoid)), xs)
D_CVaR() = DiscreteNetwork(Chain(Dense(3, 128, relu), Dense(128, 3, softplus)), xs)
    
𝒮 = CERL_Discrete(π=MixtureNetwork([D_CDF(), D_CVaR(), px], [0.45, 0.45, 0.1]), 
                  px=px,
                  priority_fn = log_err_pf,
                  α=1e-5,
                  prioritized=true,
                  pre_train_callback=compute_risk_cb(1000), 
                  log=(;period=1000), 
                  π_explore=πexplore, 
                  N=1000000,
                  c_loss=multi_td_loss(names=["Q_CDF", "Q_CVaR"], loss=Flux.Losses.msle, weight=:weight),
                  S=state_space(rmdp))
solve(𝒮, rmdp)
drl_samps, drl_weights = get_samples(𝒮.buffer, px)


## Plot the value function map
heatmap(-3:0.01:3, -3:0.01:3, (x,y)->log(value(𝒮.agent.π.networks[1], [0.1, x, y])[1]))

## Plot the distribution of costs
histogram(drl_samps, bins=0:0.1:3)
histogram!(mc_samps, bins=0:0.1:3)

# Make the plots
make_plots([mc_samps, drl_samps], [mc_weights, drl_weights], ["MC", "DRL"], 1e-5)

