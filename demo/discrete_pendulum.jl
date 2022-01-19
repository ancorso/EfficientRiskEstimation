using POMDPs, POMDPGym, POMDPSimulators, POMDPPolicies, Distributions, Plots
using Crux, Flux
using BSON
using ImportanceWeightedRiskMetrics

## Setup and solve the mdp
mdp = InvertedPendulumMDP(λcost=1, Rstep=.1, dt=dt, px=Normal(0f0, 0.1f0))
# policy = ActorCritic(GaussianPolicy(ContinuousNetwork(Chain(Dense(2, 32, relu), Dense(32, 1, tanh), x-> x .* 2f0), 1), [0f0]), 
#                      ContinuousNetwork(Chain(Dense(2, 32, relu), Dense(32, 1))))
# policy = solve(PPO(π=policy, S=state_space(mdp), N=100000, ΔN=400, max_steps=400), mdp)
# BSON.@save "policies/pendulum_policy.bson" policy

policy = BSON.load("policies/pendulum_policy.bson")[:policy]
Crux.gif(mdp, policy, "out.gif", max_steps=100,)


## Construct the risk estimation mdp where actions are disturbances
dt = 0.1
maxT = 2.0
# px = Normal(0f0, 0.35f0)
px = Normal(0f0, 1.0f0)

# cost environment
env = InvertedPendulumMDP(dt=dt, failure_thresh=Inf)
costfn(m, s, sp) = isterminal(m, sp) ? abs(sp[2]) : 0
rmdp = RMDP(env, policy, costfn, true, dt, maxT)

# Failure environment
env = InvertedPendulumMDP(dt=dt, failure_thresh=π/2)
costfn(m, s, sp) = isterminal(m, sp) ? isfailure(m, sp) : 0
rmdp = RMDP(env, policy, costfn, true, dt, maxT)


samps = [simulate(RolloutSimulator(), rmdp, FunctionPolicy((s) -> rand(px))) for _=1:10000]

m = IWRiskMetrics(samps, ones(length(samps)), 0.01)


histogram(samps, alpha=0.5, label="MC")
vline!([m.mean], label="mean", color=1)
vline!([m.var], label="var", linestyle=:dashdot, color=1)
vline!([m.cvar], label="cvar", linestyle=:dash, color=1)
vline!([m.worst], label="worst case", linestyle=:dot, color=1)

