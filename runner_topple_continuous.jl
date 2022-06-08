using BSON
include("src/environments/pendulum_topple.jl")
include("src/amis.jl")
include("src/deeprl.jl")
include("src/plotting.jl")

function plot_gmm(model, s, Npts=100)
    rd(v) = round(v, digits=2)
    αs = rd.(model.weights(s))
    m1 = rd(model.networks[1].μ(s)[1])
    σ1 = rd(exp(model.networks[1].logΣ(s)[1]))
    m2 = rd(model.networks[2].μ(s)[1])
    σ2 = rd(exp(model.networks[2].logΣ(s)[1]))
    
    lb = min(m1 - 3*σ1, m2 - 3*σ2)
    ub = max(m1 + 3*σ1, m2 + 3*σ2)
    
    yrange = range(lb, ub, length=Npts)
    py = [exp(logpdf(model, s, [y])[1]) for y in yrange]
    
    plot(yrange, py, title="$(αs[1]) ⋅ N($m1, $σ1) + $(αs[2]) ⋅ N($m2, $σ2) ")
end

function DeepGMM(idim)
    base = Chain(Dense(idim, 4, relu)) 
    mu1 = ContinuousNetwork(Chain(base..., Dense(4, 1)))
    logΣ1 = ContinuousNetwork(Chain(base..., Dense(4, 1)))
    
    mu2 = ContinuousNetwork(Chain(base..., Dense(4, 1)))
    logΣ2 = ContinuousNetwork(Chain(base..., Dense(4, 1)))
    
    αs = ContinuousNetwork(Chain(base..., Dense(4, 2), softmax), 2)
    MixtureNetwork([GaussianPolicy(mu1, logΣ1), GaussianPolicy(mu2, logΣ2)], αs)
end


α = 1e-2
Ntrials=1
N_ground_truth=10_000
N_experiment=10_000

# Setup output directory
dir = "output/pendulum_topple_continuous_α=$(α)"
try mkdir(dir) catch end

## Generate the problem setup
Px, mdp = gen_topple_mdp(px=Normal(0, 0.4), Nsteps=20, dt=0.1, failure_thresh=π/4, discrete=false)
S = state_space(mdp)
A = action_space(Px)

## Show the distributions over returns
D = episodes!(Sampler(mdp, PolicyParams(π=Px, pa=Px)), explore=true, Neps=10000)
vals = D[:r][:][D[:done][:] .== 1]
histogram(vals, label="Pendulum Topple", xlabel="Return", ylabel="Count", title="Distribution of Returns")
savefig("$dir/histogram_of_returns.png")

## Monte Carlo Approach - (ground truth experiment)
P = MDPSampler(mdp, Px)
MC_data = AMIS(N=N_ground_truth, weight_fn=(args...)->1; P, Ntrials, α, saveall=false)

save_data(dir, "MC", MC_data)
save_trajectories(mdp, Px, dir, "MC")


## Deep-AMIS
m = DeepGMM(S.dims[1])

# Train it to be nominal
y = reshape(Float32.(rand(Px.distribution, length(D[:r]))), 1, :)
d = Flux.Data.DataLoader((D[:s], y), batchsize=1024)

loss(x,y) = -mean(logpdf(m, x, y))




as = [action(m, D[:s][:,i])[1] for i=1:length(D[:r])]
plot_gmm(m, [0.1, 0.1, 0.1])
histogram!(as, alpha=0.2, normalize=true)
plot!(-3:0.1:3, x -> pdf(Px.distribution, x))



Flux.@epochs 100 Flux.train!(loss, Flux.params(m), d, ADAM())

as = [action(m, D[:s][:,i])[1] for i=1:length(D[:r])]
plot_gmm(m, [0.1, 0.1, 0.1])
histogram!(as, alpha=0.2, normalize=true)
plot!(-3:0.1:3, x -> pdf(Px.distribution, x))

m = DAIS_data[:dists][end].sampler.agent.π
μ1s = m.networks[1].μ(D[:s])
α1s = m.weights(D[:s])[1,:]
μ2s = m.networks[2].μ(D[:s])
α2s = m.weights(D[:s])[2,:]

plot(-1:0.01:1, x -> exp(logpdf(m, [0,0,0], x)[1]))
plot!(-3:0.1:3, x -> pdf(Px.distribution, x))

scatter(α1s, μ1s[:])
scatter!(α2s, μ2s[:])

#
Nsamps_per_update=10
πfn = () -> m

# Regular adaptive importance sampling
d0, update_distribution = DeepSampler(;Px, mdp, πfn, α)
DAIS_data = AMIS(N=100; weight_style=:DM, P, Ntrials, α, Nsamps_per_update, d0, update_distribution)

save_data(dir, "DAIS", DAIS_data, color=3, cdata=[MC_data], cnames=["MC"], ccolors=[1])
save_trajectories(mdp, DAIS_data[:dists][end].sampler.agent.π, dir, "DAIS", color=3, cπs=[Px], cnames=["MC"], ccolors=[1])

DAIS_data[:dists]

