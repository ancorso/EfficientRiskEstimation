using POMDPs, POMDPGym, Crux, BSON, Flux, Distributions

## Setup and solve the mdp
# dt = 0.2
# mdp = PendulumMDP(λcost=1, dt=dt)

# 
# amin = [-2f0]
# amax = [2f0]
# rand_policy = FunctionPolicy((s) -> Float32.(rand.(Uniform.(amin, amax))))
# 
# QSA() = ContinuousNetwork(Chain(Dense(3, 64, relu), Dense(64, 64, relu), Dense(64, 1)))
# A() = ContinuousNetwork(Chain(Dense(2, 64, relu), Dense(64, 64, relu), Dense(64, 1, tanh)), 1)
# 
# off_policy = (S=state_space(mdp),
#               ΔN=50,
#               N=30000,
#               buffer_size=Int(5e5),
#               buffer_init=1000,
#               c_opt=(batch_size=100, optimizer=ADAM(1e-3)),
#               a_opt=(batch_size=100, optimizer=ADAM(1e-3)),
#               π_explore=FirstExplorePolicy(1000, rand_policy, GaussianNoiseExplorationPolicy(0.5f0, a_min=[-2.0], a_max=[2.0])))
# 
# # Solver with DDPG
# 𝒮_ddpg = DDPG(;π=ActorCritic(A(), QSA()), off_policy...)
# policy = solve(𝒮_ddpg, mdp)

# BSON.@save "policies/swingup_policy.bson" policy

## Load the policy
policy = BSON.load("policies/swingup_policy.bson")[:policy]
# Crux.gif(mdp, policy, "out.gif", max_steps=20, Neps=10)
# 
# heatmap(-3:0.1:3, -8:0.1:8, (x,y) -> action(policy, [x,y])[1])


## Construct the risk estimation mdp where actions are disturbances
dt=0.2 # Do not change
maxT=3.8 # Do not change
μ=0f0
σ²=0.2f0
swingup_discrete_xs = [-0.5f0, -0.25f0, 0f0, 0.25f0, 0.5f0]

P_swingup_continuous = GaussianPolicy(ContinuousNetwork(s -> fill(μ, 1, size(s)[2:end]...), 1), ContinuousNetwork(s -> fill(Base.log.(σ²),1,size(s)[2:end]...), 1))
function discrete_logpdfs(s)
   μs = P_swingup_continuous.μ(s)
   logΣs = P_swingup_continuous.logΣ(s)
   # out = Array{Float32}(undef, length(swingup_discrete_xs), size(s)[2:end]...)
   # for i = 1:length(swingup_discrete_xs)
   #     out[i:i,:] .= 
   # end
   # out
   # 
   vcat([Crux.gaussian_logpdf(μs, logΣs, swingup_discrete_xs[i]) for i = 1:length(swingup_discrete_xs)]...)
end


P_swingup_discrete = DiscreteNetwork(s -> discrete_logpdfs(s), swingup_discrete_xs, (π,s) -> softmax(value(π,s)), true)
P_swingup_uniform = DiscreteNetwork(s -> ones(Float32, length(swingup_discrete_xs), size(s)[2:end]...), swingup_discrete_xs, (π,s) -> softmax(value(π,s)), true)


# cost environment
env = PendulumMDP(dt=dt, θ0=Uniform(1.5, 1.5000001), ω0=Uniform(0.0, 0.0000001))
costfn(m, s, sp) = isterminal(m, sp) ? abs(s[2]) : 0
swingup_mdp = RMDP(env, policy, costfn, true, dt, maxT, :arg)


