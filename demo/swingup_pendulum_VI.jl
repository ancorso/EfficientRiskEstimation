using POMDPs, POMDPGym, POMDPSimulators, POMDPPolicies, Distributions, Plots
using Crux, Flux, BSON, ImportanceWeightedRiskMetrics, Zygote
using GridInterpolations
include("utils.jl")

dt = 0.2
mdp = PendulumMDP(λcost=1, dt=dt)
policy = BSON.load("policies/swingup_policy.bson")[:policy]

## Construct the risk estimation mdp where actions are disturbances
dt=0.2 # Do not change
maxT=3.8 # Do not change
μ=0f0
σ²=0.2f0
discrete_xs = [-0.5f0, -0.25f0, 0f0, 0.25f0, 0.5f0]

px_continuous = GaussianPolicy(ContinuousNetwork(s -> fill(μ, 1, size(s)[2:end]...), 1), ContinuousNetwork(s -> fill(Base.log.(σ²),1,size(s)[2:end]...), 1))
function discrete_logpdfs(s)
   μs = px_continuous.μ(s)
   logΣs = px_continuous.logΣ(s)
   out = Array{Float32}(undef, length(discrete_xs), size(s)[2:end]...)
   for i = 1:length(discrete_xs)
       out[i:i,:] .= Crux.gaussian_logpdf(μs, logΣs, discrete_xs[i])
   end
   out
end
px_discrete = DiscreteNetwork(s -> discrete_logpdfs(s), discrete_xs, (π,s) -> softmax(value(π,s)), true)
px_uniform = DiscreteNetwork(s -> ones(Float32, length(discrete_xs), size(s)[2:end]...), discrete_xs, (π,s) -> softmax(value(π,s)), true)

# cost environment
env = PendulumMDP(dt=dt, θ0=Uniform(1.5, 1.6))
costfn(m, s, sp) = isterminal(m, sp) ? abs(s[2]) : 0
rmdp = RMDP(env, policy, costfn, true, dt, maxT)

function optimal_var_policy(mdp, target, θs=range(-π, π, length=21), ωs = range(-8, 8, length=41), ts = range(0, maxT, step=dt), as = discrete_xs)
    grid = RectangleGrid(θs, ωs, ts)

    𝒮 = [[maxT-t, θ, ω] for θ in θs, ω in ωs, t in ts]

    # State value function
    U = zeros(length(𝒮))

    # State-action value function
    Q = [zeros(length(𝒮)) for a in as]

    # Solve with backwards induction value iteration
    i=1
    for (si, s) in enumerate(𝒮)
        for (ai, a) in enumerate(as)
            s′, r = gen(mdp, s, a)
            Q[ai][si] = abs(r) > target
            Q[ai][si] += isterminal(mdp, s′) ? 0.0 : GridInterpolations.interpolate(grid, U, [maxT-s′[1], s′[2:end]...])
        end
        probs = softmax(discrete_logpdfs(s))
        U[si] = sum(p*q[si] for (q, p) in zip(Q, probs))
    end
    Q, grid
end

Q, g = optimal_var_policy(rmdp, 1.0)

q3 = Q[3]

Q

heatmap(range(-π, π, length=21), range(-8, 8, length=41), (x,y) -> GridInterpolations.interpolate(g, q3, [0.2, x, y]))

