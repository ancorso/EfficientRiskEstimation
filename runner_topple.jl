using BSON
include("src/environments/pendulum_topple.jl")
include("src/amis.jl")
include("src/deeprl.jl")
include("src/plotting.jl")

## Set the global experiment params
for α in [1e-2, 1e-3, 1e-4, 1e-5]
	println("running α=$α")
	Ntrials=10
	N_ground_truth=1_000_000
	N_experiment=10_000

	# Setup output directory
	dir = "output/pendulum_topple_α=$(α)"
	mkdir(dir)

	## Generate the problem setup
	Px, mdp = gen_topple_mdp(px=Normal(0, 0.4), Nsteps=20, dt=0.1, failure_thresh=π/4)
	S = state_space(mdp)
	A = action_space(Px)

	## Show the distributions over returns
	D = episodes!(Sampler(mdp, PolicyParams(π=Px, pa=Px)), explore=true, Neps = 10000)
	vals = D[:r][:][D[:done][:] .== 1]
	histogram(vals, label="Pendulum Topple", xlabel="Return", ylabel="Count", title="Distribution of Returns")
	savefig("$dir/histogram_of_returns.png")

	## Monte Carlo Approach - (ground truth experiment)
	P = MDPSampler(mdp, Px)
	MC_data = AMIS(N=N_ground_truth, weight_fn=(args...)->1; P, Ntrials, α, saveall=false)

	save_data(dir, "MC", MC_data)
	save_trajectories(mdp, Px, dir, "MC")

	## Uniform IS
	Pu = MDPSampler(mdp, DistributionPolicy(DiscreteNonParametric(A.vals, fill(1/A.N, A.N))))
	unif_data = AMIS(N=N_experiment,d0=Pu; P, Ntrials, α)

	save_data(dir, "UniformIS", unif_data, color=2, cdata=[MC_data], cnames=["MC"], ccolors=[1])
	save_trajectories(mdp, Pu.sampler.agent.π, dir, "UniformIS", color=2, cπs=[Px], cnames=["MC"], ccolors=[1])


	## Deep-AMIS
	Π(;kwargs...) = DiscreteNetwork(Chain(Dense(3, 32, relu), Dense(32, 32, relu), Dense(32, A.N)), Px.distribution.support, always_stochastic=true; kwargs...)
	Nsamps_per_update=200
	πfn = () -> Π(logit_conversion=policy_match_logits(Px))

	# Regular adaptive importance sampling
	d0, update_distribution = DeepSampler(;Px, mdp, πfn)
	DAIS_data = AMIS(N=N_experiment; P, Ntrials, α, Nsamps_per_update, d0, update_distribution)

	save_data(dir, "DAIS", DAIS_data, color=3, cdata=[MC_data], cnames=["MC"], ccolors=[1])
	save_trajectories(mdp, DAIS_data[1][:dists][end].sampler.agent.π, dir, "DAIS", color=3, cπs=[Px], cnames=["MC"], ccolors=[1])

	# Adaptive multiple importance sampling (using DM weights across time)
	d0, update_distribution = DeepSampler(;Px, mdp, πfn)
	DAMIS_data = AMIS(N=N_experiment, weight_style=:DM; P, Ntrials, α, Nsamps_per_update, d0, update_distribution)

	save_data(dir, "DAMIS", DAMIS_data, color=4, cdata=[MC_data], cnames=["MC"], ccolors=[1])
	save_trajectories(mdp, DAMIS_data[1][:dists][end].sampler.agent.π, dir, "DAMIS", color=4, cπs=[Px], cnames=["MC"], ccolors=[1])


	## Multiple Importance Sampling variation
	d0, update_distribution = DeepSampler(;Px, mdp, πfn)
	d0 = MISDistribution([P, Pu, d0], [1,1,8])
	MIS_DAIS_data = AMIS(N=N_experiment; P, Ntrials, α, Nsamps_per_update, d0, update_distribution)

	save_data(dir, "MIS_DAIS", MIS_DAIS_data, color=5, cdata=[MC_data], cnames=["MC"], ccolors=[1])
	save_trajectories(mdp, MIS_DAIS_data[1][:dists][end].sampler.agent.π, dir, "MIS_DAIS", color=5, cπs=[Px], cnames=["MC"], ccolors=[1])


	d0 = MISDistribution([P, Pu, d0], [1,1,8])
	d0, update_distribution = DeepSampler(;Px, mdp, πfn)
	MIS_DAMIS_data = AMIS(N=N_experiment, weight_style=:DM; P, Ntrials, α, Nsamps_per_update, d0, update_distribution)

	save_data(dir, "MIS_DAMIS", MIS_DAMIS_data, color=6, cdata=[MC_data], cnames=["MC"], ccolors=[1])
	save_trajectories(mdp, MIS_DAMIS_data[1][:dists][end].sampler.agent.π, dir, "MIS_DAMIS", color=6, cπs=[Px], cnames=["MC"], ccolors=[1])
end
  