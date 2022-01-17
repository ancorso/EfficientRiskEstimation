using POMDPs, POMDPGym, POMDPSimulators, POMDPPolicies, Distributions, Plots
using Crux, Flux
using BSON

dt = 0.1
maxT = 1.0

## Setup and solve the mdp
mdp = InvertedPendulumMDP(λcost=1, Rstep=.1, dt=dt, px=Normal(0f0, 0.1f0))
# policy = ActorCritic(GaussianPolicy(ContinuousNetwork(Chain(Dense(2, 32, relu), Dense(32, 1, tanh), x-> x .* 2f0), 1), [0f0]), 
#                      ContinuousNetwork(Chain(Dense(2, 32, relu), Dense(32, 1))))
# policy = solve(PPO(π=policy, S=state_space(mdp), N=100000, ΔN=400, max_steps=400), mdp)
# BSON.@save "policies/pendulum_policy.bson" policy

policy = BSON.load("policies/pendulum_policy.bson")[:policy]
Crux.gif(mdp, policy, "out.gif", max_steps=100,)


## Construct the risk estimation mdp where actions are disturbances
env = InvertedPendulumMDP(dt=dt, failure_thresh=Inf)
costfn(m, s, sp) = isterminal(m, sp) ? abs(sp[2]) : 0
rmdp = RMDP(env, policy, costfn, true, dt, maxT)





samps = [simulate(RolloutSimulator(), rmdp, FunctionPolicy((s) -> 0)) for _=1:1000]

histogram(samps)

