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
px_discrete = DiscreteNetwork(s -> discrete_logpdfs(s), discrete_xs, (vals,s) -> softmax(vals), true)
px_uniform = DiscreteNetwork(s -> ones(Float32, length(discrete_xs), size(s)[2:end]...), discrete_xs, (vals,s) -> softmax(vals), true)

# cost environment
env = PendulumMDP(dt=dt, θ0=Uniform(1.5, 1.6))
costfn(m, s, sp) = isterminal(m, sp) ? abs(s[2]) : 0
rmdp = RMDP(env, policy, costfn, true, dt, maxT)

struct VaR_IS_Policy <: Policy
    Q
    grid
    px
end

function Crux.exploration(π::VaR_IS_Policy, s; kwargs...)
    vals =  [GridInterpolations.interpolate(π.grid, q, [s[2], s[3], maxT-s[1]]) for q in π.Q]
    if all(vals .== 0)
        vals .= 1
    end
    probs = Crux.logits(px_discrete, s)
    
    ps = vals .* probs
    ps = ps ./ sum(ps, dims=1)
    i = rand(Categorical(ps))
    [π.px.outputs[i]], Base.log(ps[i])
end

Crux.action_space(π::VaR_IS_Policy) = action_space(π.px)
Crux.new_ep_reset!(π::VaR_IS_Policy) = nothing


function optimal_var_policy(mdp, target, θs=range(-π, π, length=200), ωs = range(-8, 8, length=200), ts = range(0, maxT, step=dt), as = discrete_xs)
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
            Q[ai][si] += isterminal(mdp, s′) ? 0.0 : GridInterpolations.interpolate(grid, U,[ s′[2:end]..., maxT-s′[1]])
        end
        probs = softmax(discrete_logpdfs(s))
        U[si] = sum(p*q[si] for (q, p) in zip(Q, probs))
    end
    VaR_IS_Policy(Q, grid, px_discrete)
end

D = episodes!(Sampler(rmdp, px_discrete, required_columns=[:logprob]), Neps=1000, explore=true)
samps, weights = get_samples(ExperienceBuffer(D), px_discrete)
target = 0.7
for i=1:10
    
    println("target: ", target)
    
    pol = optimal_var_policy(rmdp, 0.7)
    D = episodes!(Sampler(rmdp, pol, required_columns=[:logprob]), Neps=1000, explore=true)
    new_samps, new_weights = get_samples(ExperienceBuffer(D), px_discrete)
    
    push!(samps, new_samps...)
    push!(weights, new_weights...)
    
    risk_metrics = IWRiskMetrics(samps, weights, 0.001)
    target = risk_metrics.var
end

risk_metrics = IWRiskMetrics(samps, weights, 0.001)
target = risk_metrics.var
println("target: ", target)

    










histogram(samps)




h1 = heatmap(range(-π, π, length=21), range(-8, 8, length=41), (x,y) -> GridInterpolations.interpolate(pol.grid, pol.Q[1], [x, y, maxT-2.2]), clims=(0,1))
h2 = heatmap(range(-π, π, length=21), range(-8, 8, length=41), (x,y) -> GridInterpolations.interpolate(pol.grid, pol.Q[2], [x, y, maxT-2.2]), clims=(0,1))
h3 = heatmap(range(-π, π, length=21), range(-8, 8, length=41), (x,y) -> GridInterpolations.interpolate(pol.grid, pol.Q[3], [x, y, maxT-2.2]), clims=(0,1))
h4 = heatmap(range(-π, π, length=21), range(-8, 8, length=41), (x,y) -> GridInterpolations.interpolate(pol.grid, pol.Q[4], [x, y, maxT-2.2]), clims=(0,1))
h5 = heatmap(range(-π, π, length=21), range(-8, 8, length=41), (x,y) -> GridInterpolations.interpolate(pol.grid, pol.Q[5], [x, y, maxT-2.2]), clims=(0,1))


plot(h1, h2, h3, h4, h5)



