using POMDPs, POMDPGym, POMDPSimulators, POMDPPolicies, Distributions, Plots
using Crux, Flux, BSON, Random, StaticArrays



# Basic MDP
tprob = 0.7
Random.seed!(0)
randcosts = Dict(POMDPGym.GWPos(i,j) => rand() for i = 1:10, j=1:10)
mdp = GridWorldMDP(costs=randcosts, cost_penalty=0.1, tprob=tprob)

render(mdp) # The reward map
render(mdp, color = s->10.0*POMDPGym.cost(mdp, s)) # The cost map

# Learn a policy that solves it
# policy = DiscreteNetwork(Chain(x -> (x .- 5f0) ./ 5f0, Dense(2, 32, relu), Dense(32, 4)), [:up, :down, :left, :right])
# policy = solve(DQN(π=policy, S=state_space(mdp), N=100000, ΔN=4, buffer_size=10000, log=(;period=5000)), mdp)
# atable = Dict(s => action(policy, [s...]) for s in states(mdp.g))
# BSON.@save "demo/gridworld_policy_table.bson" atable

atable = BSON.load("demo/gridworld_policy_table.bson")[:atable]

# Define the adversarial mdp
adv_rewards = deepcopy(randcosts)
for (k,v) in mdp.g.rewards
    if v < 0
        adv_rewards[k] += -10*v
    end
end

amdp = GridWorldMDP(rewards=adv_rewards, tprob=1., discount=1.)

render(amdp, color = s->log(100*POMDPs.reward(amdp.g, s)))

# Define action probability for the adv_mdp
action_probability(mdp, s, a) = (a == atable[s]) ? tprob : ((1. - tprob) / (length(actions(mdp)) - 1.))

