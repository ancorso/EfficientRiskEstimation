using POMDPs, POMDPGym, POMDPSimulators, POMDPPolicies, Distributions, Plots
using Crux, Flux, BSON, ImportanceWeightedRiskMetrics
using POMDPModelTools
include("utils.jl")

## Setup and solve the mdp
dt = 0.1
mdp = InvertedPendulumMDP(λcost=1, Rstep=1, dt=dt, px=Normal(0f0, 0.1f0))
# policy = ActorCritic(GaussianPolicy(ContinuousNetwork(Chain(Dense(2, 32, relu), Dense(32, 1, tanh), x-> x .* 1f0), 1), [0f0]), 
#                      ContinuousNetwork(Chain(Dense(2, 32, relu), Dense(32, 1))))
# policy = solve(PPO(π=policy, S=state_space(mdp), N=200000, ΔN=500, max_steps=100), mdp)
# BSON.@save "policies/pendulum_policy.bson" policy

policy = BSON.load("policies/pendulum_policy.bson")[:policy]
Crux.gif(mdp, policy, "out.gif", max_steps=100,)

heatmap(-3:0.1:3, -8:0.1:8, (x,y) -> action(policy, [x,y])[1])


## Construct the risk estimation mdp where actions are disturbances
dt=0.1 # Do not change
maxT=2.0 # Do not change
μ=0f0
σ²=0.1f0
discrete_xs = [-1.5f0, -1f0, -0.5f0, 0f0, 0.5f0, 1f0, 1.5f0]

px_continuous = GaussianPolicy(ContinuousNetwork(s -> fill(μ, 1, size(s)[2:end]...), 1), 
                               ContinuousNetwork(s -> reshape(Base.log.(σ² .* (1f0 .+ pdf.(Normal(0.5, 0.15), s[1,:]) )),1,size(s)[2:end]...), 1))
function discrete_logpdfs(s)
   μs = px_continuous.μ(s)
   logΣs = px_continuous.logΣ(s)
   out = Array{Float32}(undef, length(discrete_xs), size(s)[2:end]...)
   for i = 1:length(discrete_xs)
       out[i:i,:] .= Crux.gaussian_logpdf(μs, logΣs, discrete_xs[i])
   end
   out
end
px_discrete = DiscreteNetwork(s -> discrete_logpdfs(s), discrete_xs, (vals, s)->softmax(vals), true)
px_uniform = DiscreteNetwork(s -> ones(Float32, length(discrete_xs), size(s)[2:end]...), discrete_xs, (vals,s) -> softmax(vals), true)
px_zeros = DiscreteNetwork(s -> ones(Float32, length(discrete_xs), size(s)[2:end]...), zeros(size(discrete_xs)), (vals,s) -> softmax(vals), true)




# cost environment
env = InvertedPendulumMDP(dt=dt, failure_thresh=Inf, θ0=Uniform(-0.1f0, 0.1f0),ω0=Uniform(-0.2f0, 0.2f0))
costfn(m, s, sp) = isterminal(m, sp) ? abs(sp[2]) : 0
rmdp = RMDP(env, policy, costfn, true, dt, maxT)

## Make some plots of mc
ps, px = plot(), plot()
p = plot_pendulum(rmdp, px_discrete, 1, Neps=1000, ps=ps, px=px)

## Get ground trutch estimates
mc_samps = [simulate(RolloutSimulator(), rmdp, FunctionPolicy((s) -> exploration(px_discrete, s)[1][1])) for _=1:Int(1e4)]
# BSON.@save "data/500thou_mcsamps_discretependulum_$(σ²).bson" mc_samps 
histogram(mc_samps, ylims=(0,1))

# mc_samps = BSON.load("data/500thou_mcsamps_discretependulum_$(σ²).bson")[:mc_samps]
mc_weights = ones(length(mc_samps))

# compute risk metrics
risk_metrics = IWRiskMetrics(mc_samps, mc_weights, 0.001)
risk_metrics.cvar
risk_metrics.var
risk_metrics.mean
risk_metrics.worst

## Setup and run deep rl approach
function estimator_logits(vals, s)
    probs = Crux.logits(px_discrete, s)
    ps = vals .* probs
    ps ./ sum(ps, dims=1)
end
D_CDF() = DiscreteNetwork(Chain(Dense(3, 64, relu), Dense(64,64,relu), Dense(64, length(discrete_xs), sigmoid)), discrete_xs, estimator_logits, true) # Low priority, hyperparameter
D_CVaR() = DiscreteNetwork(Chain(Dense(3, 64, relu), Dense(64,64,relu), Dense(64, length(discrete_xs), sigmoid), x->x.*3.15f0), discrete_xs, estimator_logits, true) # Low priority, hyperparameter
    
𝒮 = ISDRL_Discrete(π=MixtureNetwork([D_CDF(), D_CVaR(), px_discrete], [0.5, 0.5, 0.0]), 
                  px=px_discrete,
                  priority_fn=log_err_pf, #log_err_pf, abs_err_pf
                  α=1e-3, # experiment parameter
                  prioritized=true, # false <- ablation
                  use_likelihood_weights=false, #false (hyperparameter)
                  pre_train_callback=compute_risk_cb(1000, 0.1), # [0.01, 0.1, 0.5] 0.1 is the minmum fraction of samples in tail. (hyperparameter)
                  log=(;period=1000),
                  # π_explore=MixedPolicy(Crux.LinearDecaySchedule(1.0, 0.0, 20000), px_uniform),
                  N=200000, # Number of steps in the environment
                  c_loss=multi_td_loss(names=["Q_CDF", "Q_CVaR"], loss=Flux.Losses.msle, weight=:weight), # if prioritized, then weight=nothing. loss can be: [Flux.Losses.msle, Flux.Losses.mse] <- ablation
                  S=state_space(rmdp))

solve(𝒮, rmdp)
drl_samps, drl_weights = get_samples(𝒮.buffer, px_discrete)

risk_metrics = IWRiskMetrics(drl_samps, drl_weights, 0.001)

plot(risk_metrics.Z)
plot(risk_metrics.est)

plot(risk_metrics.est.Xs)


cvar_weights = drl_weights[drl_samps .>=  𝒮.𝒫.rα[1]]
cvar_vals = drl_samps[drl_samps .>=  𝒮.𝒫.rα[1]]


plot(log.(drl_weights))
plot(drl_samps)

risk_metrics = IWRiskMetrics(drl_samps[drl_weights .< 0.1], drl_weights[drl_weights .< 0.1], 0.001)


ids = drl_samps .>= risk_metrics.var

cvar_weights = drl_weights[drl_samps .>= risk_metrics.var]

sum(cvar_weights)^2 / (sum(cvar_weights .^ 2))

cvar_samples = drl_samps[drl_samps .>= risk_metrics.var]
histogram(cvar_samples)

sum(cvar_samples .* (cvar_weights ./ sum(cvar_weights)))

mean(cvar_samples)

histogram(cvar_samples)

plot(log.(drl_weights))
plot(drl_samps)

D = episodes!(Sampler(rmdp, PolicyParams(π=𝒮.agent.π.networks[1], pa=px_discrete), required_columns=[:logprob, :likelihoodweight]), Neps=10, explore=true)

sum(D[:r] .> 0.48)

# plot(D[:s][1,:], D[:s][2,:], marker=true)
plot(D[:s][1,:], discrete_xs[Flux.onecold(D[:a])], marker=true)
plot(D[:s][1,:], D[:likelihoodweight][:], marker=true)


D = episodes!(Sampler(rmdp, px_discrete), Neps=10000, explore=true)
is = findall(D[:r][:] .> 0.48)
p = plot()
for i in is
    irange = i-19:i
# plot(D[:s][1,irange], D[:sp][2,irange], marker=true)
    plot!(p, D[:s][1,irange], discrete_xs[Flux.onecold(D[:a][:,irange])], marker=true)
end
p

plot(D[:s][1,irange], D[:likelihoodweight][:], marker=true)




plot_pendulum(rmdp, 𝒮.agent.π, 3, Neps=100, ps=ps, px=px)
hline(ps, [-𝒮.𝒫.rα[1]], label="VaR")

## Solve with CEM
function cem_loss_weight(d, xin)
    x = xin[:x]
    s = rand(initialstate(rmdp))
    rtot=0.0
    logweight=0.0
    for i=1:length(x)
        @assert !isterminal(rmdp, s)
        logweight += logpdf(px_discrete, s, [x[i]])[1]
        s, r = gen(rmdp, s, x[i])
        rtot += r
    end
    @assert isterminal(rmdp, s)
    return rtot, exp(logweight - logpdf(d,xin))
end


d0 = Dict{Symbol, Tuple{Sampleable, Int64}}(:x => (DiscreteNonParametric(discrete_xs, ones(Float32, length(discrete_xs)) ./ length(discrete_xs)), 21))
vals, weights, dopt = cem(cem_loss_weight, d0, max_iter=10, N=1000, α=1e-3)


cem_policy = DistributionPolicy(dopt[1][:x][1])




p = plot!([], color=4, label="CEM")
plot_pendulum(rmdp, cem_policy, 4, ps=ps, px=px, Neps=1000)




#TODO: Compute error to ground truth abs(gt - est) / gt

plot(log.(drl_weights))
plot(drl_samps)

## Plot the value function map
heatmap(-3:0.01:3, -8:0.01:8, (x,y)->value(𝒮.agent.π.networks[1], [1.9, x, y])[3], xlabel="θ", ylabel="ω") # cdf 
heatmap(-3:0.01:3, -8:0.01:8, (x,y)->value(𝒮.agent.π.networks[2], [1.9, x, y])[3], xlabel="θ", ylabel="ω")

indices = 𝒮.buffer[:s][1,:] .≈ 1.9
scatter!(𝒮.buffer[:s][2,indices], 𝒮.buffer[:s][3,indices], color = Int.(abs.(𝒮.buffer[:s][2,indices]) .> 𝒮.𝒫.rα[1]))

𝒮.buffer[:s]

y = 𝒮.target_fn(𝒮.agent.π, 𝒮.𝒫, 𝒮.buffer, 1f0)[1]
y[𝒮.buffer[:done][:]]
r = 𝒮.buffer[:r][1,𝒮.buffer[:done][:]]

indices =  .≈ 2.0
sum(indices)

𝒮.target_fn

## Plot the distribution of costs
histogram(drl_samps, bins=0:0.1:3)
histogram!(mc_samps, bins=0:0.1:3)

# Make the plots
make_plots([mc_samps, drl_samps], [mc_weights, drl_weights], ["MC", "DRL"], 1e-5)

