using POMDPs, POMDPGym, POMDPSimulators, POMDPPolicies, Distributions, Plots
using Crux, Flux, BSON, Random, StaticArrays
using ImportanceWeightedRiskMetrics
Crux.set_function("isfailure", POMDPGym.isfailure)
include("discrete_exploration.jl")
include("utils.jl")

# Basic MDP
tprob = 0.7
Random.seed!(0)
randcosts = Dict(POMDPGym.GWPos(i,j) => rand() for i = 1:10, j=1:10)
mdp = GridWorldMDP(costs=randcosts, cost_penalty=0.1, tprob=tprob)

# Learn a policy that solves it
# policy = DiscreteNetwork(Chain(x -> (x .- 5f0) ./ 5f0, Dense(2, 32, relu), Dense(32, 4)), [:up, :down, :left, :right])
# policy = solve(DQN(π=policy, S=state_space(mdp), N=100000, ΔN=4, buffer_size=10000, log=(;period=5000)), mdp)
# atable = Dict(s => action(policy, [s...]) for s in states(mdp.g))
# BSON.@save "demo/gridworld_policy_table.bson" atable

atable = BSON.load("demo/gridworld_policy_table.bson")[:atable]

render(mdp) # The reward map
render(mdp, color = s->10.0*POMDPGym.cost(mdp, s)) # The cost map
render(mdp.g, policy = FunctionPolicy(s->atable[s][1]))


# Define the adversarial mdp
adv_rewards = Dict{GWPos, Float64}()
for (k,v) in mdp.g.rewards
    if v < 0
        adv_rewards[k] = 1
    else
        adv_rewards[k] = 0
    end
end

amdp = GridWorldMDP(rewards=adv_rewards, tprob=1., discount=1.)

# Define action probability for the adv_mdp
struct GridworldDisturbance
    xs
    atable
    tprob
end

gridworld_action_probability(s, a, tprob, atable, xs) = Float32((a == px.atable[GWPos(s)][1]) ? px.tprob : ((1. - px.tprob) / (length(xs) - 1.)))

function Distributions.logpdf(px::GridworldDisturbance, s)
    Float32.(log.([gridworld_action_probability(s[:, i], x, px.tprob, px.atable, px.xs) for x in px.xs, i=1:size(s,2)]))
end

function Base.rand(px::GridworldDisturbance, s)
    if rand() < px.tprob
        return atable[s][1]
    else
        return rand(px.xs)
    end
end

px = GridworldDisturbance(actions(amdp), atable, tprob)
πexplore = DiscreteExploration((s) -> [gridworld_action_probability(s, a, px.tprob, px.atable, px.xs) for a in actions(amdp)])
pol = DiscreteNetwork(Chain(x -> (x .- 5f0) ./5f0, Dense(2, 128, relu), Dense(128, 4, sigmoid)), actions(amdp))

S = state_space(amdp)
𝒮 = CERL_Discrete(π=pol, px=px, π_explore=πexplore, N=100000, S=S, required_columns=[:logprob])
solve(𝒮, amdp)

drl_samps, drl_weights = get_samples(𝒮.buffer, px)

mc_samps = Float64.([simulate(RolloutSimulator(), amdp, FunctionPolicy((s) -> rand(px, s))) > 0.0 for _=1:100000])
mc_weights = ones(length(mc_samps))

make_plots([mc_samps, drl_samps], [mc_weights, drl_weights], ["MC", "DRL"], 0.1)



heatmap(1:10, 1:10, (x,y) -> log(value(pol, [x,y], [1,0,0,0])[1]))
heatmap(1:10, 1:10, (x,y) -> value(pol, [x,y], [0,1,0,0])[1])
heatmap(1:10, 1:10, (x,y) -> value(pol, [x,y], [0,0,1,0])[1])
heatmap(1:10, 1:10, (x,y) -> value(pol, [x,y], [0,0,0,1])[1])


render(amdp, color=(s)->10.0*value(pol, [s...], [1,0,0,0])[1])
render(amdp, color=(s)->10.0*value(pol, [s...], [0,1,0,0])[1])
render(amdp, color=(s)->10.0*value(pol, [s...], [0,0,1,0])[1])
render(amdp, color=(s)->10.0*value(pol, [s...], [0,0,0,1])[1])





