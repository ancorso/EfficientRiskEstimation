using POMDPs, POMDPGym, POMDPSimulators, POMDPPolicies, Distributions, Plots
using Crux, Flux

# Basic MDP
mdp = InvertedPendulumMDP(λcost=0, include_time_in_state=true)

# Learn a policy that solves it
policy = ActorCritic(GaussianPolicy(ContinuousNetwork(Chain(Dense(3, 32, relu), Dense(32, 1))), [0f0]), 
                     ContinuousNetwork(Chain(Dense(3, 32, relu), Dense(32, 1))))
policy = solve(PPO(π=policy, S=state_space(mdp), N=20000, ΔN=400), mdp)

# Construct the adversarial MDP to get access to a transition function like gen(mdp, s, a, x)
px = Normal(0f0, 0.5f0)
amdp = AdditiveAdversarialMDP(mdp, px)

# Construct the risk estimation mdp where actions are disturbances
rmdp = RMDP(amdp, policy, (m, s) -> 1 / abs(s[1] - mdp.failure_thresh))

rmdp.cost_fn(mdp, [3,4,5])

reward(rmdp, rand(initialstate(rmdp)))


samps = [maximum(collect(simulate(HistoryRecorder(), rmdp, FunctionPolicy((s) -> rand(px)))[:r])) for _ in 1:10000]

histogram(samps)


# Construct the MDP

