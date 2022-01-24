using CSV, DataFrames
using POMDPs, POMDPGym, POMDPSimulators, POMDPPolicies, Distributions, Plots
using Crux, Flux, BSON, ImportanceWeightedRiskMetrics

include("utils.jl")
include("discrete_exploration.jl")

output_dir = "/Users/kykim/Desktop" # Change as needed.

policy = BSON.load("policies/pendulum_policy.bson")[:policy]

# Construct the risk estimation mdp where actions are disturbances
dt = 0.1 # Do not change
maxT = 2.0 # Do not change
px_nom = Normal(0f0, 0.35f0) # Do not change

xs = [-1, 0, 1] # Do not change
pxs = pdf.(px_nom, xs)
pxs = pxs ./ sum(pxs)
px = DistributionPolicy(DiscreteNonParametric(xs, pxs))

# Cost environment
env = InvertedPendulumMDP(dt=dt, failure_thresh=Inf)
costfn(m, s, sp) = isterminal(m, sp) ? abs(sp[2]) : 0
rmdp = RMDP(env, policy, costfn, true, dt, maxT)

# Get ground trutch estimates
mc_samps = BSON.load("data/10mil_mcsamps.bson")[:mc_samps]
mc_weights = ones(length(mc_samps))

# Setup and run deep rl approach
πexplore = DiscreteExploration((s) -> pxs)
D_CDF() = DiscreteNetwork(Chain(Dense(3, 128, relu), Dense(128, 3, sigmoid)), xs) # Low priority, hyperparameter
D_CVaR() = DiscreteNetwork(Chain(Dense(3, 128, relu), Dense(128, 3, softplus)), xs) # Low priority, hyperparameter
    

function run_drl(α, priority_fn, use_likelihood_weights, min_samples, N)
    𝒮 = CERL_Discrete(π=MixtureNetwork([D_CDF(), D_CVaR(), px], [0.45, 0.45, 0.1]), 
                      px=px,
                      priority_fn=priority_fn,
                      α=α,
                      prioritized=true,
                      use_likelihood_weights=use_likelihood_weights,
                      pre_train_callback=compute_risk_cb(1000, min_samples),
                      log=(;period=10_000_000), 
                      π_explore=πexplore, 
                      N=N,
                      c_loss=multi_td_loss(names=["Q_CDF", "Q_CVaR"], loss=Flux.Losses.msle, weight=:weight),
                      S=state_space(rmdp))
    solve(𝒮, rmdp)
    drl_samps, drl_weights = get_samples(𝒮.buffer, px)
    return drl_samps, drl_weights
end


# Compute the error and save the results out to a csv.
function compute_error(α, mc_samps, mc_weights, drl_samps, drl_weights)
    ground_truth = IWRiskMetrics(mc_samps, mc_weights, α)

    # Take subsets of the samples and compute the min error.
    min_var_rel, min_cvar_rel = Inf, Inf
    l = length(drl_samps)
    p = l ÷ 10
    for n in p:p:l
        drl_risk_metrics = IWRiskMetrics(drl_samps[1:n], drl_weights[1:n], α)
        var_rel_err = abs(ground_truth.var - drl_risk_metrics.var) / ground_truth.var
        cvar_rel_err = abs(ground_truth.cvar - drl_risk_metrics.cvar) / ground_truth.cvar
        min_var_rel = min(min_var_rel, var_rel_err)
        min_cvar_rel = min(min_cvar_rel, cvar_rel_err)
    end
    return min_var_rel, min_cvar_rel
end


# Iterate over all params and do run_drl()
α_l = [1e-2, 1e-3, 1e-4, 1e-5]
priority_fn_l = [log_err_pf, abs_err_pf]
use_likelihood_weights_l = [true, false]
min_samples_l = [0.01, 0.1, 0.5]
N_l = [10_000, 100_000]

df = DataFrame(alpha=Float64[], priority_fn=String[], use_lw=String[], min_sample=Float64[], N=Integer[],
               var_err=Float64[], cvar_err=Float64[])

all_params = [p for p in Iterators.product(α_l, priority_fn_l, use_likelihood_weights_l, min_samples_l, N_l)]
for (idx, params) in enumerate(all_params)
    println("Running ", idx, " / ", length(all_params))
    α, priority_fn, use_lw, min_samples, N = params
    try
        drl_samps, drl_weights = run_drl(α, priority_fn, use_lw, min_samples, N)
        min_var_rel, min_cvar_rel = compute_error(α, mc_samps, mc_weights, drl_samps, drl_weights)
        global df
        push!(df,
              [α,
               priority_fn == log_err_pf ? "log_err_pf" : "abs_err_pf",
               use_lw ? "true" : "false",
               min_samples,
               N,
               min_var_rel, min_cvar_rel]);
        if idx % 20 == 0
            CSV.write(string(output_dir, "/drl_dp_", idx, ".csv"), df)
        end
    catch e
        println("Exception thrown for ", params, ": ", e)
        continue
    end
end

CSV.write(string(output_dir, "/drl_dp.csv", df)
