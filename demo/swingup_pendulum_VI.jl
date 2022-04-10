using POMDPs, POMDPGym, POMDPSimulators, POMDPPolicies, Distributions, Plots
using Crux, Flux, BSON, ImportanceWeightedRiskMetrics, Zygote
using GridInterpolations
using Printf
include("utils.jl")

dt = 0.2
mdp = PendulumMDP(λcost=1, dt=dt)
policy = BSON.load("policies/swingup_policy.bson")[:policy]
# Crux.gif(mdp, policy, "out.gif", max_steps=20, Neps=10)
# heatmap(-3:0.1:3, -8:0.1:8, (x,y) -> action(policy, [x,y])[1], title="Swingup Policy", xlabel="θ", ylabel="ω")

plot(rand(10))


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

## Make disturbance plot historgram
# xs = [action(px_discrete, [1,1,1.])[1] for _=1:10000]
# histogram(xs, title="Disturbances", xlabel="torque", ylabel="count", label="")

function intermeditate_estimates(samps, weights, Nsamps = Int.(floor.(10 .^ range(2, log10(length(samps)), length=10))); Nbootstrap=100, α)
    results = Dict()
    for r in Nsamps
        println("running for nsteps: $r")
        vars, cvars = Float64[], Float64[]
        for i=1:Nbootstrap
            indices = rand(1:length(samps), r)
            risk_metrics = IWRiskMetrics(samps[indices], weights[indices], α, 0)
            push!(vars, risk_metrics.var)
            push!(cvars, risk_metrics.cvar)
            # var_std = std(risk_metrics.bootstrap_vars)
            # cvar_std = std(risk_metrics.bootstrap_cvars)
        end
        var = mean(vars)
        cvar = mean(cvars)
        var_std = std(vars)
        cvar_std = std(cvars)
        results[r] = (;var, cvar, var_std, cvar_std)
    end
    results
end

function make_var_convergence_plot(samps, weights, Nsamps = Int.(floor.(10 .^ range(2, log10(length(samps)), length=10))); Nbootstrap=100, α=0.001, title="", p=plot(title=title, xlabel="No. Samples", ylabel="VaR"))
    vars, cvars = [], []
    vars_std, cvars_std = [], []
    for r in Nsamps
        println("running for nsteps: $r")
        risk_metrics = IWRiskMetrics(samps[1:r], weights[1:r], α, Nbootstrap)
        push!(vars, risk_metrics.var)
        push!(cvars, risk_metrics.cvar)
        push!(vars_std, std(risk_metrics.bootstrap_vars))
        push!(cvars_std, std(risk_metrics.bootstrap_cvars))
    end
    plot!(p, Nsamps, vars, label="VaR", xscale=:log10)
    plot!(p, Nsamps, vars_mean, ribbon=vars_std, label="VaR - Bootstrap")
end

## Make histogram plot
function make_VaR_histogram(samps, weights; title="", Nbootstrap=100, α=0.001)
    risk_metrics = IWRiskMetrics(samps, weights, α, Nbootstrap)

    histogram(samps, bins=0:0.01:3, label="Return Samples", title=title, xlabel="Return", ylabel="Count", normalize=true)
    histogram!(risk_metrics.bootstrap_vars, xlims=(0,3), label="VaR Bootstrap Samples", xlabel="Return", normalize=true, alpha=0.4)
    vline!([risk_metrics.var], label = @sprintf("VaR estimate: %5.3f", risk_metrics.var))
    vline!([mean(risk_metrics.bootstrap_vars)], label = @sprintf("Bootstrap VaR estimate: %5.3f", mean(risk_metrics.bootstrap_vars)))
end

function make_cdf_plots(samps, weights, chunk_size; label="", α=0.001, Nbootstrap=1, p=plot(), color=2)
    N = length(samps)
    Nchunks = floor(Int, N / chunk_size)
    samp_chunks = [samps[(i-1)*chunk_size + 1:i*chunk_size] for i=1:Nchunks]
    weight_chunks = [weights[(i-1)*chunk_size + 1:i*chunk_size] for i=1:Nchunks]
    
    risk_metrics = IWRiskMetrics(samps, weights, α, Nbootstrap)

    plot!(p, reverse(risk_metrics.est.Xs), risk_metrics.est.partial_Ws ./ N, label=string(label, " -- Full cdf"),  yscale=:log10, title="CDFs -- $(N) vs $(chunk_size)")
    for i=1:min(100, length(samp_chunks))
        rm = IWRiskMetrics(samp_chunks[i], weight_chunks[i], α, Nbootstrap)
        N = length(rm.Z)
        plot!(p, reverse(rm.est.Xs), rm.est.partial_Ws ./ N, label="", alpha=0.2, color=color, yscale=:log10)
    end
    plot!(p, [], color=color, label=string(label, " -- Chunk size: $(chunk_size)"))
end



## Ground truth var
# mc_samps = [simulate(RolloutSimulator(), rmdp, FunctionPolicy((s) -> exploration(px_discrete, s)[1][1])) for _=1:Int(5e7)]
# BSON.@save "data/10mil_mcsamps_discreteswingup_$(σ²).bson" mc_samps 

mc_samps = BSON.load("data/10mil_mcsamps_discreteswingup_$(σ²).bson")[:mc_samps]
mc_weights = ones(length(mc_samps))

# dict = Dict()
# for α = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
#     println("Computing Risk metrics for α: $α")
#     rm = IWRiskMetrics(mc_samps, mc_weights, α, 10)
# 
#     dict[α] = (var=rm.var, bootstrap_vars=rm.bootstrap_vars, cvar=rm.cvar, bootstrap_cvars=rm.bootstrap_cvars)
# end
# 
# BSON.@save "data/10mil_groundtruth_rms.bson" dict

# make_var_convergence_plot(mc_samps, mc_weights, title="Monte Carlo Estimate of VaR")
# make_VaR_histogram(mc_samps, mc_weights, title="1 Million MC Samples")
# 
# p1k = make_cdf_plots(mc_samps, mc_weights, 1000, label="MC")
# p10k = make_cdf_plots(mc_samps, mc_weights, 10000, label="MC")
# p100k = make_cdf_plots(mc_samps, mc_weights, 100000, label="MC")
# 
# plot(p1k, p10k, p100k, layout=(1,3), size=(1800,400))
# savefig("mc.png")

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


## Attempt 1 
# pol = optimal_var_policy(rmdp, risk_metrics.var_cdf)
# D = episodes!(Sampler(rmdp, pol, required_columns=[:logprob]), Neps=1000000, explore=true)
# new_samps, new_weights = get_samples(ExperienceBuffer(D), px_discrete)
# data = (;new_samps, new_weights)
# BSON.@save "data/1mil_issamps_discreteswingup_0.2_attempt1.bson" data

# new_samps, new_weights = BSON.load("data/1mil_issamps_discreteswingup_0.2_attempt1.bson")[:data]
# 
# is_rm = IWRiskMetrics(new_samps, new_weights, 0.001, 1)
# 
# # Make initial plots
# make_var_convergence_plot(new_samps, new_weights, title="IS Estimate of VaR")
# make_VaR_histogram(new_samps, new_weights, title="1 Million IS Samples")
# 
# isp1k = make_cdf_plots(new_samps, new_weights, 1000, label="IS", p=p1k, color=3)
# isp10k = make_cdf_plots(new_samps, new_weights, 10000, label="IS", p=p10k, color=3)
# isp100k = make_cdf_plots(new_samps, new_weights, 100000, label="IS", p=p100k, color=3)
# 
# plot(isp1k, isp10k, isp100k, layout=(1,3), size=(1800,400))
# savefig("is.png")


## Attempt 2 
# pol = optimal_var_policy(rmdp, x->mc_risk_metrics.var_cdf(x+0.5))
# D = episodes!(Sampler(rmdp, pol, required_columns=[:logprob]), Neps=1000000, explore=true)
# new_samps, new_weights = get_samples(ExperienceBuffer(D), px_discrete)
# data = (;new_samps, new_weights)
# BSON.@save "data/1mil_issamps_discreteswingup_0.2_attempt2.bson" data
# 
# new_samps, new_weights = BSON.load("data/1mil_issamps_discreteswingup_0.2_attempt2.bson")[:data]
# 
# # Make initial plots
# make_var_convergence_plot(new_samps, new_weights, title="IS Estimate of VaR")
# make_VaR_histogram(new_samps, new_weights, title="1 Million IS Samples")
# 
# isp1k = make_cdf_plots(new_samps, new_weights, 1000, label="IS", p=p1k, color=3)
# isp10k = make_cdf_plots(new_samps, new_weights, 10000, label="IS", p=p10k, color=3)
# isp100k = make_cdf_plots(new_samps, new_weights, 100000, label="IS", p=p100k, color=3)
# 
# plot(isp1k, isp10k, isp100k, layout=(1,3), size=(1800,400))
# savefig("is.png")

## range of targets 
# for target in 0.1:0.1:3.0
#     println("running $target")
#     pol = optimal_var_policy(rmdp, x->x > target)
#     D = episodes!(Sampler(rmdp, pol, required_columns=[:logprob]), Neps=Int(1e6), explore=true)
#     new_samps, new_weights = get_samples(ExperienceBuffer(D), px_discrete)
#     data = (;new_samps, new_weights)
#     BSON.@save "data/1mil_issamps_discreteswingup_target_$(target).bson" data
# end

## Get all results
mc_samps_1m = mc_samps[1:Int(1e6)]
mc_weights_1m = ones(Int(1e6))
# 
# 
all_results = Dict()
for α = [1e-3, 1e-4, 1e-5]
    println("======= α=$α")
    mcresults = intermeditate_estimates([mc_samps_1m..., mc_samps_1m...], [mc_weights_1m..., mc_weights_1m...], α=α)
    Nsamps = keys(mcresults)
    for r in Nsamps
        all_results[(α, 0.0, r)] = mcresults[r]
    end
    for target in [0.1, 0.7, 1.3, 1.9, 2.5]
        issamps, isweights = BSON.load("data/1mil_issamps_discreteswingup_target_$(target).bson")[:data]
        isresults = intermeditate_estimates([issamps..., mc_samps_1m...], [isweights...,mc_weights_1m...], α=α, Nbootstrap=10)
        @assert keys(isresults) == Nsamps

        for r in Nsamps
            all_results[(α, target, r)] = isresults[r]
        end
    end
end

BSON.@save "data/is_mc_mixed_results.bson" all_results
    


all_results = BSON.load("data/is_mc_mixed_results.bson")[:all_results]
ground_truth = BSON.load("data/10mil_groundtruth_rms.bson")[:dict]

function mdsi_keys(dict)
    ks = keys(dict)
    all_keys = []
    for i=1:length(first(ks))
        push!(all_keys, sort(unique([k[i] for k in ks])))
    end
    return all_keys
end

function  mdsi_slice(dict, slice, sym)
    ks = mdsi_keys(dict)

    slice[slice .== Colon()] .= ks[slice .== Colon()]
    slice[.! isa.(slice, AbstractArray)] 

    Ns = (length.(slice)...,)
    arr = Array{Float64, length(Ns)}(undef, Ns...)
    for ijk in CartesianIndices(Ns)
        key = ([slice[i][j] for (i,j) in zip(1:length(Ns), ijk.I)]...,)
        arr[ijk] = dict[key][sym]
    end
    return dropdims(arr, dims = tuple(findall(size(arr) .== 1)...))
end

for (sym, sym_std) in zip([:var, :cvar], [:var_std, :cvar_std])
    for α in mdsi_keys(all_results)[1]
        Nsamps = mdsi_keys(all_results)[3]

        plots = []
        for target in [0.1, 0.7, 1.3, 1.9, 2.5] #0.1:0.1:3.0
            p = plot(Nsamps, fill(ground_truth[α][sym], length(Nsamps)), ribbon=fill(std(ground_truth[α][:bootstrap_vars]), length(Nsamps)), xscale=:log10, label="", title=@sprintf("%s, target: %5.3f, gt: %5.3f", sym, target, ground_truth[α][sym])) 
            plot!(Nsamps, mdsi_slice(all_results, [α, 0.0, Nsamps], sym), ribbon=mdsi_slice(all_results, [α, 0.0, Nsamps], sym_std), label = "")
            plot!(Nsamps, mdsi_slice(all_results, [α, target, Nsamps], sym), ribbon=mdsi_slice(all_results, [α, target, Nsamps], sym_std), label="")
            push!(plots, p)
        end
        plot(plots..., layout=(5,1), size= (floor(Int, 2000/6), 2000))
        savefig("$(sym)_alpha_$(α).png")
    end
end



target = 0.6
plot!(Nsamps, mdsi_slice(all_results, [α, target, Nsamps], :var), ribbon=mdsi_slice(all_results, [α, target, Nsamps], :var_std), label = "IS - target: $(target)")

target = 0.8
plot!(Nsamps, mdsi_slice(all_results, [α, target, Nsamps], :var), ribbon=mdsi_slice(all_results, [α, target, Nsamps], :var_std), label = "IS - target: $(target)")

target = 1.0
plot!(Nsamps, mdsi_slice(all_results, [α, target, Nsamps], :var), ribbon=mdsi_slice(all_results, [α, target, Nsamps], :var_std), label = "IS - target: $(target)")

target = 2.2
plot!(Nsamps, mdsi_slice(all_results, [α, target, Nsamps], :var), ribbon=mdsi_slice(all_results, [α, target, Nsamps], :var_std), label = "IS - target: $(target)")




new_samps, new_weights = BSON.load("data/1mil_issamps_discreteswingup_0.2_attempt1.bson")[:data]
is_rm = IWRiskMetrics(new_samps[1:10000], new_weights[1:10000], 0.001, 100)

is_rm.var

histogram(new_samps[1:1000])

est = ImportanceWeightedRiskMetrics.bootstrap_VaR_cdf(new_samps[1:1000], new_weights[1:1000], 0.001, 100, return_ests=true)

make_cdf_plots(est[1].Xs, est[1].Ws, 1000)


p = plot(yscale=:log10)
for i=1:100
    N = 1000
    plot!(p, reverse(est[i].Xs), est[i].partial_Ws ./ N, label="", alpha=0.3)
end
p
    







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



