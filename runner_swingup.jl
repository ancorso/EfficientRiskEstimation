include("src/environments/pendulum_swingup.jl")
include("src/amis.jl")
include("src/deeprl.jl")
include("src/plotting.jl")


α=0.001
S = state_space(swingup_mdp)
A = action_space(P_swingup_discrete)
Puniform = MDPSampler(swingup_mdp, P_swingup_uniform)


## Monte Carlo Approach
P = MDPSampler(swingup_mdp, P_swingup_discrete)

mc_data = AMIS(N=10000, P=P, weight_fn=(args...)->1, Ntrials=3, α=α)

# Plot the CDF
pcdf = plot_cdf(mc_data[1], α=α, label="MC", xlabel="f", ylabel="1-cdf(f)")
pcdf = plot_cdf(mc_data[2], α=α, label="MC - 2", p=pcdf)

# Plot the convergence
plt = plot_convergence(mc_data, label="MC", title="VaR Convergence", xlabel="Samples", ylabel="VaR")



## AMIS
# function estimator_logits(π, s)
#     vals = value(π, s)
#     probs = Crux.logits(P_swingup_discrete, s)
#     ps = vals .* probs
#     ps ./ sum(ps, dims=1)
# end

# model() = DiscreteNetwork(Chain(Dense(3, 32, relu), Dense(32, length(swingup_discrete_xs), sigmoid)), swingup_discrete_xs, estimator_logits, true)
model() = DiscreteNetwork(Chain(Dense(3, 32, relu), Dense(32,32,relu), Dense(32,32,relu),Dense(32, length(swingup_discrete_xs))), swingup_discrete_xs, (π, s)->softmax(value(π, s)), true)
V() = ContinuousNetwork(Chain(Dense(3, 32, relu), Dense(32, 1)))



# Regular AMIS
gen_d() = MDPSampler(swingup_mdp, model())
N = 10000
Nsamps_per_update = 200
Nepochs = 1000 #Nsamps_per_update

amis_std = AMIS(;N, d0=gen_d(), P, update_distribution=trainer(S, A, N*20, epochs=Nepochs), Ntrials=1, α, weight_style=:standard, Nsamps_per_update, log_period=Nsamps_per_update, log_period_multiplier=1)
# amis_dm = AMIS(;N, d0=gen_d(), P, update_distribution=trainer(S, A, N*20, epochs=Nepochs), Ntrials=3, α, weight_style=:DM, Nsamps_per_update, log_period=Nsamps_per_update, log_period_multiplier=1)

solver = REINFORCE(π=, S=state_space(swingup_mdp), N=200000, ΔN=2048)
solve(solver, swingup_mdp)

function var_loss(π, 𝒫, 𝒟; info = Dict())
    new_probs = logpdf(π, 𝒟[:s], 𝒟[:a])
    nom_probs = logpdf(P.π, 𝒟[:s], 𝒟[:a])
    e_loss = -mean(entropy(π, 𝒟[:s]))
    
    Flux.Zygote.ignore() do
        info[:entropy] = -e_loss
        info[:kl] = mean(𝒟[:logprob] .- new_probs)
    end 
    
    -mean(new_probs .* (𝒟[:return] .> 0.5f0) .* exp.(nom_probs .- new_probs)) + 
end

solver = OnPolicySolver(;agent=PolicyParams(model()),
                S=state_space(swingup_mdp), N=900000, ΔN=20000,
                log = LoggerParams(;dir = "log/pfail"),
                a_opt = TrainingParams(;loss = var_loss, name = "actor_", optimizer=ADAM(1e-3)),
                required_columns = unique([ :return, :logprob]))

pol = solve(solver, swingup_mdp)
pol = solver.agent.π

sampler =  MDPSampler(swingup_mdp, pol)

fs = Float64[]
xs = []
ws = Float64[]
for i=1:10000
    f, x = rand(sampler)
    w = pdf(P, x) / pdf(sampler, x)
    push!(ws, w) 
    push!(xs, x) 
    push!(fs, f) 
end

f, x = rand(sampler)
w = pdf(P, x) / pdf(sampler, x)

Crux.logits(pol, x[:s])


rm = IWRiskMetrics(fs, ws, 0.001)
rm.var


solver = A2C(π=ActorCritic(model(), V()), S=state_space(swingup_mdp), N=200000, ΔN=2048)
solve(solver, swingup_mdp)

solver = PPO(π=ActorCritic(model(), V()), S=state_space(swingup_mdp), N=200000, λe=0f0)
solve(solver, swingup_mdp)


solver = DQN(π=model(), S=state_space(swingup_mdp), N=200000, π_explore=ϵGreedyPolicy(LinearDecaySchedule(1,1,20000), swingup_discrete_xs))
solve(solver, swingup_mdp)

# Defensive
gen_mis_d() = MISDistribution([P, Puniform, gen_d()], [1,1,8])

mis_amis_std_10 = AMIS(;N, d0=gen_mis_d(), P, update_distribution=trainer(S, A, N*20, epochs=Nepochs), Ntrials=3, α, weight_style=:standard, Nsamps_per_update, log_period=Nsamps_per_update, log_period_multiplier=1)
mis_amis_dm_10 = AMIS(;N, d0=gen_mis_d(), P, update_distribution=trainer(S, A, N*20, epochs=Nepochs), Ntrials=3, α, weight_style=:DM, Nsamps_per_update, log_period=Nsamps_per_update, log_period_multiplier=1)


plot_cdf(amis_std[1], α=α)
plot(log.(amis_std[1][:ws]))
plot(amis_std[1][:fs], xlabel="sample", ylabel="cost", label="", title="Samples from Deep AMIS")

plt = plot_convergence(mcdata, label="MC", title="VaR Convergence", xlabel="Samples", ylabel="VaR")
plot_convergence(amis_std, p=plt, label="Deep AMIS - Standard Weighting")
plot_convergence(amis_dm, p=plt, label="Deep AMIS - DM Weighting")
plot_convergence(mis_amis_std_10, p=plt, label="Deep Defensive AMIS - Standard Weighting")
plot_convergence(mis_amis_dm_10, p=plt, label="Deep Defensive AMIS - DM Weighting")


results = (amis_std, amis_dm, mis_amis_std_10, mis_amis_dm_10)
BSON.@save "tempresults.bson" results

d = data[:dists][end]

xs = []
fs = Float64[]
ws = Float64[]
for i=1:1000
    fv, x = rand(d)
    w = pdf(P, x) / pdf(mis_d, x)
    push!(xs, x)
    push!(fs, fv)
    push!(ws, w)
end

histogram(Flux.onecold(hcat([d[:a] for d in xs]...)))

histogram(fs)

histogram(log.(ws))

plot_cdf((;xs, fs, ws, var=data[:var]), α=α)

Crux.logits(d.distributions[2].π, xs[end][:s])

