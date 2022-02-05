using POMDPs, POMDPGym, POMDPSimulators, POMDPPolicies, Distributions, Plots
using Crux, Flux, BSON, ImportanceWeightedRiskMetrics
include("utils.jl")
include("cem.jl")

## Setup and solve the mdp
dt = 0.2
mdp = PendulumMDP(λcost=1, dt=dt)
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
Crux.gif(mdp, policy, "out.gif", max_steps=20, Neps=10)
# 
# heatmap(-3:0.1:3, -8:0.1:8, (x,y) -> action(policy, [x,y])[1])


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

##
ps = plot()
px = plot()

## Make some plots of mc
mc_samps = [simulate(RolloutSimulator(), rmdp, FunctionPolicy((s) -> exploration(px_discrete, s)[1][1])) for _=1:Int(1e4)]

histogram(mc_samps, ylims=(0,1), xlims=(0,3))

# BSON.@save "data/200thou_mcsamps_discreteswingup_$(σ²).bson" mc_samps 

# mc_samps = BSON.load("data/200thou_mcsamps_discreteswingup_$(σ²).bson")[:mc_samps]
mc_weights = ones(length(mc_samps))
histogram(mc_samps, ylims=(0,1))

# compute risk metrics
risk_metrics = IWRiskMetrics(mc_samps_orig, mc_weights, 0.001)
risk_metrics.cvar
risk_metrics.var
risk_metrics.mean
risk_metrics.worst

plot_pendulum(rmdp, px_discrete, 1, Neps=100, ps=ps, px=px, label="MC - discrete")
plot_pendulum(rmdp, px_continuous, 2, Neps=1000, px=px, ps=ps, label="MC - continuous")

## solve with cross entropy
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


d0 = Dict{Symbol, Tuple{Sampleable, Int64}}(:x => (DiscreteNonParametric(discrete_xs, ones(Float32, length(discrete_xs)) ./ length(discrete_xs)), 20))
vals, weights, dopt = cem(cem_loss_weight, d0, max_iter=10, N=1000, α=1e-3)

histogram(vals, ylims=(0,100))
histogram(log.(weights), ylims=(0,100))

histogram(log.(drl_weights))

cem_policy = DistributionPolicy(dopt[1][:x][1])
plot_pendulum(rmdp, cem_policy, 3, Neps=100, px=px, ps=ps, label="CEM")

plot(ps, legend=:bottomleft)


## Setup and run deep rl approach
function cdf_td_loss(loss)
    (π, 𝒫, 𝒟, y; info=Dict()) -> begin
        s = Zygote.ignore() do
            B = length(𝒟[:r])
            z = reshape(repeat(𝒫[:rs], 1, B)', 1, :)
            s = repeat(𝒟[:s], 1, length(𝒫[:rs]))
            vcat(z, s)
        end
        Q = value(critic(π), s, 𝒟[:a])
        
        # Store useful information
        ignore() do
            info[name] = mean(Q)
        end
        
        loss(Q, y)
    end
end

function estimator_logits(vals, s)
    probs = Crux.logits(px_discrete, s)
    ps = vals .* probs
    ps ./ sum(ps, dims=1)
end

D_CDF() = LatentConditionedNetwork(DiscreteNetwork(Chain(Dense(4, 64, tanh), Dense(64, 64, tanh), Dense(64, length(discrete_xs), sigmoid)), discrete_xs, estimator_logits, true), [0f0])
# D_CVaR() = DiscreteNetwork(Chain(Dense(3, 64, tanh), Dense(64, 64, tanh), Dense(64, length(discrete_xs), sigmoid), x->x.*3.15f0), discrete_xs, estimator_logits, true)
N_cdf=10
𝒮 = ISDRL_Discrete(π=D_CDF()), 
                  px=px_discrete,
                  priority_fn=abs_err_pf, #log_err_pf, abs_err_pf
                  ΔN=20,
                  N_cdf=N_cdf,
                  α=1e-3, # experiment parameter
                  prioritized=true, # false <- ablation
                  use_likelihood_weights=false, #false (hyperparameter)
                  pre_train_callback=compute_risk_cb(1000, 0.1, N_cdf=N_cdf), # [0.01, 0.1, 0.5] 0.1 is the minmum fraction of samples in tail. (hyperparameter)
                  log=(;period=1000),
                  # c_opt=(;batch_size=512),
                  π_explore=MixedPolicy(Crux.LinearDecaySchedule(0.3, 0.02, 200000), px_uniform), 
                  N=200000, # Number of steps in the environment
                  c_loss=cdf_td_loss(loss=Flux.Losses.msle), # if prioritized, then weight=nothing. loss can be: [Flux.Losses.msle, Flux.Losses.mse] <- ablation
                  S=state_space(rmdp))
solve(𝒮, rmdp)

Dnom = episodes!(Sampler(rmdp, px_discrete, required_columns=[:logprob]), Neps=10000, explore=true)


sum(Dnom[:r] .> 1.2)
elite = findall(Dnom[:r][:] .> 1.2)
epranges = hcat([collect(e-19:e) for e in elite]...)
eprangesv = vcat([collect(e-19:e) for e in elite]...)
plot(Dnom[:s][1, epranges], Dnom[:s][2, epranges], label="")

plot(Dnom[:s][1, epranges], discrete_xs[Flux.onecold(Dnom[:a][:, epranges])], label="")


D = episodes!(Sampler(rmdp, PolicyParams(π=𝒮.agent.π.networks[1], pa=px_discrete), required_columns=[:logprob, :likelihoodweight, :var_prob, :cvar_prob]), Neps=10, explore=true)

sum(D[:r]) / 10

plot(reshape(D[:s][1, :], 20, :), reshape(discrete_xs[Flux.onecold(D[:a][:, :])],  20, :), label="")

pmine = logpdf(𝒮.agent.π.networks[1], Dnom[:s][:, eprangesv], Dnom[:a][:, eprangesv])

sum(pmine)

pnom = logpdf(px_discrete, D[:s], D[:a])
pmine = D[:logprob]

pnom = Dnom[:logprob][1,eprangesv]
sum(pnom)

plot(pmine[:])
plot!(pnom[:])
plot!(D[:likelihoodweight][:])

minimum(D[:likelihoodweight][:])
.022*.022

Dnom[:s][:, eprangesv]
# What is the logprob of the most common failures from rejection sampling, and what is the logprob
# of failures for the drl policy?


histogram(log.(𝒮.buffer[:likelihoodweight][:]))

drl_samps, drl_weights = get_samples(𝒮.buffer, px_discrete)

plot(log.(drl_weights))


D = episodes!(Sampler(rmdp, PolicyParams(π=𝒮.agent.π.networks[1], pa=px_discrete), required_columns=[:logprob, :likelihoodweight, :var_prob, :cvar_prob]), Neps=1, explore=true)

plot(D[:s][1,:], D[:s][2,:])
plot(D[:s][1,:], discrete_xs[Flux.onecold(D[:a])], label="action")
plot!(D[:s][1,:], D[:likelihoodweight][1,:], label="likelihood")

prod(D[:likelihoodweight][1,:])
vals = value(𝒮.agent.π.networks[1], D[:s])

for i=1:21
    println("i: ",i, " likelihood: ", D[:likelihoodweight][i], " vals: ", vals[:,i])
end
D[:likelihoodweight]

𝒮.𝒫.rα

T = 13
s1, r = gen(rmdp, D[:s][:,T], discrete_xs[1])
s2, r = gen(rmdp, D[:s][:,T], discrete_xs[2])
s3, r = gen(rmdp, D[:s][:,T], discrete_xs[3])
s4, r = gen(rmdp, D[:s][:,T], discrete_xs[4])
s5, r = gen(rmdp, D[:s][:,T], discrete_xs[5])

x1 = [simulate(RolloutSimulator(), rmdp, FunctionPolicy((s) -> exploration(px_discrete, s)[1][1]), Float32.(s1)) for _=1:Int(1e4)]
x2 = [simulate(RolloutSimulator(), rmdp, FunctionPolicy((s) -> exploration(px_discrete, s)[1][1]), Float32.(s2)) for _=1:Int(1e4)]
x3 = [simulate(RolloutSimulator(), rmdp, FunctionPolicy((s) -> exploration(px_discrete, s)[1][1]), Float32.(s3)) for _=1:Int(1e4)]
x4 = [simulate(RolloutSimulator(), rmdp, FunctionPolicy((s) -> exploration(px_discrete, s)[1][1]), Float32.(s4)) for _=1:Int(1e4)]
x5 = [simulate(RolloutSimulator(), rmdp, FunctionPolicy((s) -> exploration(px_discrete, s)[1][1]), Float32.(s5)) for _=1:Int(1e4)]


v = [mean(x1 .> 𝒮.𝒫.rα), mean(x2 .> 𝒮.𝒫.rα), mean(x3 .> 𝒮.𝒫.rα), mean(x4 .> 𝒮.𝒫.rα), mean(x5 .> 𝒮.𝒫.rα)]

vps = (v .* ps) ./ sum(v .* ps)



Q = Float32[4.0491473f-6, 1.4583379f-6, 2.3946643f-6, 1.7916544f-5, 0.006859343]

ps = Crux.logits(px_discrete, rand(3))

Qps = (Q .* ps) ./ sum(Q .* ps)

bar(vps, alpha=0.2)
bar!(Qps, alpha=0.2)

ps[5] / vps[5]
ps[5] / Qps[5]


drl_risk_metrics = IWRiskMetrics(drl_samps[drl_weights .<= 1], drl_weights[drl_weights .<= 1], 0.001)

plot(reverse(drl_risk_metrics.est.Xs))
plot(drl_risk_metrics.est.partial_Ws) #, ylims=(0,5))
plot(drl_risk_metrics.est.partial_XWs)
plot(drl_risk_metrics.w[end-100:end])
plot(drl_risk_metrics.w)

drl_risk_metrics.w[end-100:end]

histogram(log.(drl_risk_metrics.w))

VaR(drl_risk_metrics.est, 0.001)

histogram(drl_samps)
histogram!(mc_samps)

drl_risk_metrics.est.last_i * .001

drl_risk_metrics.est.last_i



plot_pendulum(rmdp, 𝒮.agent.π, 5, Neps=100, px=px, ps=ps, label="DRL")
hline(ps, [-risk_metrics.var, risk_metrics.var], label="VaR", linestyle=:dash)
plot(ps, legend=:bottomleft)




plot!([], color=2, label="DRL")
𝒮.agent.π.weights = [0.5, 0.5]
p = plot_episodes(𝒮.agent.π, 2, p, π_explore=𝒮.agent.π_explore)

risk_metrics.cvar

p = plot!([], color=4, label="CEM")
plot_episodes(cem_policy, 4, p, Neps=1000)




#TODO: Compute error to ground truth abs(gt - est) / gt

plot(log.(drl_weights))
plot(drl_samps)

## Plot the value function map
heatmap(-3:0.01:3, -8:0.01:8, (x,y)->log(value(𝒮.agent.π.networks[1], [1.2, x, y])[3]), xlabel="θ", ylabel="ω") # cdf 
heatmap(-3:0.01:3, -8:0.01:8, (x,y)->value(𝒮.agent.π.networks[2], [3.8, x, y])[2], xlabel="θ", ylabel="ω")

indices = 𝒮.buffer[:s][1,:] .≈ 3.8
scatter!(𝒮.buffer[:s][2,indices], 𝒮.buffer[:s][3,indices], color = Int.(abs.(𝒮.buffer[:s][2,indices]) .> 𝒮.𝒫.rα[1]))

plot(𝒮.buffer[:var_prob][1,indices], Float32.(abs.(𝒮.buffer[:s][2,indices]) .>= 𝒮.𝒫.rα[1]))



𝒮.𝒫.rα[1]

𝒮.buffer[:s][3,indices]


epies = episodes(𝒮.buffer)
ps = plot()
px = plot()
i=1
for e in epies
    ir = e[1]:e[2]
    plot!(ps, D[:s][1, ir], sin.(D[:s][2, ir] .+ π), color=i, label="", alpha=0.5)
    if size(D[:a], 1) > 1
        plot!(px, D[:s][1, ir], discrete_xs[Flux.onecold(D[:a][:, ir])], color=i, label="", alpha=0.5)
    else
        plot!(px, D[:s][1, ir], D[:a][1, ir], color=i, label="", alpha=0.5)
    end
end
plot(ps, px, layout=(2, 1))


## Plot the distribution of costs
histogram(drl_samps, bins=0:0.1:3)
histogram!(mc_samps, bins=0:0.1:3)

# Make the plots
make_plots([mc_samps, drl_samps], [mc_weights, drl_weights], ["MC", "DRL"], 1e-5)

