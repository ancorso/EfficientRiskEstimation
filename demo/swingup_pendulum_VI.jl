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


function optimal_var_policy(mdp, cdf, θs=range(-π, π, length=100), ωs = range(-8, 8, length=400), ts = range(0, maxT, step=dt), as = discrete_xs)
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
            Q[ai][si] = cdf(abs(r))
            Q[ai][si] += isterminal(mdp, s′) ? 0.0 : GridInterpolations.interpolate(grid, U,[ s′[2:end]..., maxT-s′[1]])
        end
        probs = softmax(discrete_logpdfs(s))
        U[si] = sum(p*q[si] for (q, p) in zip(Q, probs))
    end
    VaR_IS_Policy(Q, grid, px_discrete)
end

## Run a sample and produce figures




α = 0.001
D = episodes!(Sampler(rmdp, px_discrete, required_columns=[:logprob]), Neps=1000, explore=true)
samps, weights = get_samples(ExperienceBuffer(D), px_discrete)
rm = IWRiskMetrics(samps, weights, α, 10)

# plot(0:0.01:1, rm.var_cdf)

samp_segments = [deepcopy(samps)]
weight_segments = [deepcopy(weights)]
data_segments = [D]
pol_segments = Any[px_discrete]


for i=1:10
    println("mean VaR: ", rm.var, " std VaR: ", std(rm.bootstrap_vars))
    
    pol = optimal_var_policy(rmdp, rm.var_cdf)
    push!(pol_segments, deepcopy(pol))
    
    D = episodes!(Sampler(rmdp, pol, required_columns=[:logprob]), Neps=1000, explore=true)
    push!(data_segments, D)
    # D = episodes!(Sampler(rmdp, px_discrete, required_columns=[:logprob]), Neps=1000, explore=true)
    new_samps, new_weights = get_samples(ExperienceBuffer(D), px_discrete)
    
    push!(samps, new_samps...)
    push!(weights, new_weights...)
    
    push!(samp_segments, new_samps)
    push!(weight_segments, new_weights)
    
    rm = IWRiskMetrics(samps, weights, α)
end


histogram(vcat(samp_segments...), alpha=.4, xlims=(0,3))
histogram!(mc_samps[1:10000], alpha=0.4,xlims=(0,3))

histogram(samp_segments[1], xlims=(0,3))
histogram!(samp_segments[2], xlims=(0,3))
histogram!(samp_segments[3], xlims=(0,3))
samp_segments


N=4
allsamps = vcat(samp_segments[1:10]...)
allweights = vcat(weight_segments[1:10]...)
risk_metrics = IWRiskMetrics(allsamps, allweights, 0.001)
target = risk_metrics.var
std(risk_metrics.bootstrap_vars)
mean(risk_metrics.bootstrap_vars)


findfirst(risk_metrics.est.partial_Ws .> 1)
plot(log.(reverse(risk_metrics.est.Ws)))
plot(risk_metrics.est.partial_Ws, ylims=(0,12), xlims=(1,1000))
plot!(reverse(risk_metrics.est.Xs))
hline!([10])

length(mc_samps)*0.001

rm_mc = IWRiskMetrics(mc_samps, mc_weights, 0.001)

plot(rm_mc.est.partial_Ws ./ 1000, ylims=(0,2), xlims=(1, 2000))
plot!(reverse(rm_mc.est.Xs))
hline!([1])


2000*0.001

risk_metrics = IWRiskMetrics(samps, weights, 0.001)

println("target: ", target)

is_approach = [0.7664387822151184, 0.8079857230186462, 0.798484742641449, 0.706789493560791, 0.6963788866996765, 1.172311782836914, 1.0821504592895508, 0.7228874564170837]

mc_approach = [0.9229497909545898, 0.8891160488128662, 0.49549400806427, 0.6479496359825134, 0.7082074284553528, 1.0292009115219116, 0.6223886609077454]


histogram(is_approach, label="IS", alpha=0.4, bins=0:0.1:1.5)
histogram(mc_approach, label="MC",  alpha=0.4,bins=0:0.1:1.5)
histogram!(risk_metrics.bootstrap_vars, label="GT bootstrap",  alpha=0.4,bins=0:0.1:1.5)


savefig("MC_vs_GT.pdf")








histogram(samps)





h1 = heatmap(range(-π, π, length=21), range(-8, 8, length=41), (x,y) -> GridInterpolations.interpolate(pol.grid, pol.Q[1], [x, y, maxT-2.2]), clims=(0,1))
h2 = heatmap(range(-π, π, length=21), range(-8, 8, length=41), (x,y) -> GridInterpolations.interpolate(pol.grid, pol.Q[2], [x, y, maxT-2.2]), clims=(0,1))
h3 = heatmap(range(-π, π, length=21), range(-8, 8, length=41), (x,y) -> GridInterpolations.interpolate(pol.grid, pol.Q[3], [x, y, maxT-2.2]), clims=(0,1))
h4 = heatmap(range(-π, π, length=21), range(-8, 8, length=41), (x,y) -> GridInterpolations.interpolate(pol.grid, pol.Q[4], [x, y, maxT-2.2]), clims=(0,1))
h5 = heatmap(range(-π, π, length=21), range(-8, 8, length=41), (x,y) -> GridInterpolations.interpolate(pol.grid, pol.Q[5], [x, y, maxT-2.2]), clims=(0,1))


plot(h1, h2, h3, h4, h5)



