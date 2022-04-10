using ImportanceWeightedRiskMetrics
using Distributions
using Plots
using Random
using Flux

include("src/amis.jl")

## Define the function to estimate
f(x) = (min(2^x, 1000.)) * (x > 4)
P = Normal(0,2)
p(x) = pdf(P, x)
α=1e-3


# Plot the function and what we wish to approximate
xticks = -8:0.1:10
plot(xticks, f, label="f =   2ˣ ⋅ 1{x > 4}", title="Integral to approximate", xlabel="x", ylims=(0,1), legend=:topleft)
plot!(xticks, p, label="p(x) = N(x; 0,1)")
plot!(xticks, (x)->f(x)*p(x), label="f ⋅ p")

## Run MC
mc_data = AMIS(N=1000000, P=P, weight_fn=(args...)->1, sampler=default_sampler(f), Ntrials=10, α=α)

# Plot the CDF
pcdf = plot_cdf(mc_data[1], α=α, label="MC", xlabel="f", ylabel="1-cdf(f)")

# Plot the convergence
plt = plot_convergence(mc_data, label="MC", title="MC VaR Convergence", xlabel="Samples", ylabel="VaR")

## Run IS
is1 = truncated(P, 4, Inf)
is2 = Normal(5, 0.9)
is3 = Normal(6, .2)

# Show the distributions
plot(xticks, (x)->f(x)*p(x), label="f ⋅ p", title="Importance Sampling Proposals", xlabel="x", ylims=(0,2), legend=:topleft)
plot!(xticks, (x) -> pdf(is1, x), label="IS1 - $(is1)")
plot!(xticks, (x) -> pdf(is2, x), label="IS2 - $(is2)")
plot!(xticks, (x) -> pdf(is3, x), label="IS3 - $(is3)")

# Collect the data
Nsamps = 1e5
isdata1 = AMIS(N=Nsamps, d0=is1, P=P, sampler=default_sampler(f), Ntrials=10, α=α)
isdata2 = AMIS(N=Nsamps, d0=is2, P=P, sampler=default_sampler(f), Ntrials=10, α=α)
isdata3 = AMIS(N=Nsamps, d0=is3, P=P, sampler=default_sampler(f), Ntrials=10, α=α)

# Plot cdfs
pcdf = plot_cdf(mc_data[1], α=α, label="MC")
plot_cdf(isdata1[1], α=α, p=pcdf, label="IS1")
plot_cdf(isdata2[1], α=α, p=pcdf, label="IS2")
plot_cdf(isdata3[1], α=α, p=pcdf, label="IS3")

# Plot convergence
plt = plot_convergence(mc_data, label="MC", title="IS estimate of VaR (10 trials)", xlabel="Samples", ylabel="VaR")
plot_convergence(isdata1, label="IS1 - $(is1)", p=plt)
plot_convergence(isdata2, label="IS2 - $(is2)", p=plt)
plot_convergence(isdata3, label="IS3 - $(is3)", p=plt)


## Run defensive importance sampling
defensive_dist = MISDistribution([P, is3])

# Show the distributions
plot(xticks, (x)->f(x)*p(x), label="f ⋅ p", title="Defensive Importance Sampling Proposals", xlabel="x", ylims=(0,1), legend=:topleft)
plot!(xticks, x->pdf(is3,x), label="IS3")
plot!(xticks, p, color=3, label="Nominal")

# Collect the data
defensive_data = AMIS(N=Nsamps, d0=defensive_dist, P=P, sampler=default_sampler(f), Ntrials=10, α=α)

# Plot cdfs
pcdf = plot_cdf(mc_data[1], α=α, label="MC")
plot_cdf(isdata3[1], α=α, p=pcdf, label="IS3")
plot_cdf(defensive_data[1], α=α, p=pcdf, label="Defensive")

# Plot Convergence
plt = plot_convergence(mc_data, label="MC", title="Defensive IS estimate of VaR (10 trials)", xlabel="Samples", ylabel="VaR")
plot_convergence(isdata3, label="IS3 - Normal(6, .2)", p=plt)
plot_convergence(defensive_data, Nsamps=Nsamps, label="defensive IS", p=plt)


## Run  Multiple Importance Sampling
MIS_dists = [Normal(3,.2), Normal(4,.2), Normal(5,.2), Normal(6,.2),  Normal(7,.2),  Normal(8,.2)]
MIS_dist_standard = MISDistribution(MIS_dists, weight_style=:standard)
MIS_dist_dm = MISDistribution(MIS_dists, weight_style=:DM)

mis_p = plot(xticks, (x)->f(x)*p(x), label="f ⋅ p", title="Multiple Importance Sampling Proposals", xlabel="x", ylims=(0,2), legend=:topleft)
for d in MIS_dists
	plot!(xticks, x->pdf(d,x), color=3, label=d==first(MIS_dists) ? "MIS dists" : "")
end
mis_p

# Collect the data
mis_data_standard = AMIS(N=Nsamps, d0=MIS_dist_standard, P=P, sampler=default_sampler(f), Ntrials=10, α=α)
mis_data_dm = AMIS(N=Nsamps, d0=MIS_dist_dm, P=P, sampler=default_sampler(f), Ntrials=10, α=α)

# Plot convergence
plt = plot_convergence(mc_data, label="MC", title="Multiple IS estimate of VaR (10 trials)", xlabel="Samples", ylabel="VaR")
plot_convergence(mis_data_standard, label="MIS - Standard Weights", p=plt)
plot_convergence(mis_data_dm, label="MIS - DM Weights", p=plt)


## Adaptive MIS
d0 = Normal(0,.5)
Nsamps = 2e4
Nsamps_per_update = 500

function fp_prop(;xs, fs, ws, α, kwargs...)
	fit(Normal, Float64.(xs), (ws .+ 1e-6) .* (fs .+ 1e-6))
end

function var_prop(;xs, fs, ws, α, kwargs...)
	Nsamps = length(fs)
	risk_metric = IWRiskMetrics(fs, ws, 1.1*α)
	var_min = sort(fs, rev=true)[10]
	
	target = min(risk_metric.var, var_min)
	
	fit(Normal, Float64.(xs), (ws .+ 1e-6) .* (fs .>= target))
end

amis_data_standard = AMIS(N=Nsamps; d0, P, Nsamps_per_update, update_distribution=fp_prop, sampler=default_sampler(f), Ntrials=10, α, weight_style=:standard)
amis_data_dm = AMIS(N=Nsamps; d0, P, Nsamps_per_update, update_distribution=fp_prop, sampler=default_sampler(f), Ntrials=10, α, weight_style=:DM)
amis_data_dm_var = AMIS(N=Nsamps; d0, P, Nsamps_per_update, update_distribution=var_prop, sampler=default_sampler(f), Ntrials=10, α, weight_style=:DM)

plt = plot_convergence(mc_data, label="MC", title="Adaptive MIS estimate of VaR (10 trials)", xlabel="Samples", ylabel="VaR")
plot_convergence(amis_data_standard, label="AMIS - Standard Weights - fp fit", p=plt)
plot_convergence(amis_data_dm, label="AMIS - DM Weights - fp fit", p=plt)
plot_convergence(amis_data_dm_var, label="AMIS - DM Weights - var fit", p=plt)


plt = plot_pdfs(xticks, amis_data_standard[1][:dists], color=:algae, label="standard")
plot_pdfs(xticks, amis_data_dm[1][:dists], color=:dense, label="dm", p=plt)

plt = plot_pdfs(xticks, amis_data_dm_var[1][:dists], color=:algae, label="dm - varfit")



## Deep-adaptive multiple importance sampling
NN_model() = Chain(Dense(3, 32, relu), Dense(32, 2))

function Distributions.pdf(d::Chain, x)
	out = d(ones(3))
	dist = Normal(out[1], exp(out[2]))
	pdf(dist, x)
end

function Distributions.logpdf(d::Chain, x)
	out = d(ones(3))
	Crux.gaussian_logpdf(out[1], out[2], x)
end

# Define the sampler
function Base.rand(rng::AbstractRNG, d::Chain)
	out = d(ones(3))
	dist = Normal(out[1], exp(out[2]))
	rand(rng, dist)
end

function train_model(;d, xs, fs, ws, α, all_weights, kwargs...)
	risk_metric = IWRiskMetrics(fs, ws, 1.1*α)
	var_min = sort(fs, rev=true)[10]
	
	target = min(risk_metric.var, var_min)
	
	weights  = reshape((ws .+ 1e-6) .* (fs .>= target), 1, :)
	x = reshape(deepcopy(xs), 1, :)
	
	loss() = -mean(logpdf(d, x) .* weights)
	θ = Flux.params(d)
	opt = ADAM(1e-3)
	
	# println("training with $(length(xs))...")
	last_loss = Inf
	for i=1:100
		train_loss, back = Flux.pullback(loss, θ)
   		gs = back(one(train_loss))
   		Flux.update!(opt, θ, gs)
		
		if abs((last_loss - train_loss) / train_loss) < 0.001
			# println("break after $i iterations")
			break
		end
		last_loss = train_loss
	end
	
	d
end


damis_data_standard = AMIS(N=Nsamps; d0=NN_model(), P, Nsamps_per_update, update_distribution=train_model, sampler=default_sampler(f), Ntrials=10, α, weight_style=:standard)
damis_data_dm = AMIS(N=Nsamps; d0=NN_model(), P, Nsamps_per_update, update_distribution=train_model, sampler=default_sampler(f), Ntrials=10, α, weight_style=:DM)


plt = plot_convergence(mc_data, label="MC", title="Adaptive MIS estimate of VaR (10 trials)", xlabel="Samples", ylabel="VaR")
plot_convergence(amis_data_standard, label="deep AMIS - Standard Weights - var fit", p=plt)
plot_convergence(amis_data_dm, label="deep AMIS - DM Weights - var fit", p=plt)



## (Dropout) Deep-adaptive multiple importance sampling
dropout_model() = Chain(Dense(3, 32, relu), Dropout(0.1),  Dense(32, 2))

function Distributions.pdf(d::Chain, x)
	trainmode!(d)
	ps = []
	for i=1:10
		out = d(ones(3))
		dist = Normal(out[1], exp(out[2]))
		push!(ps, pdf(dist, x))
	end
	return mean(ps)
end


# Define the sampler
function Base.rand(rng::AbstractRNG, d::Chain)
	testmode!(d)
	out = d(ones(3))
	dist = Normal(out[1], exp(out[2]))
	rand(rng, dist)
end


damis_dropout_data_standard = AMIS(N=Nsamps; d0=dropout_model(), P, Nsamps_per_update, update_distribution=train_model, sampler=default_sampler(f), Ntrials=10, α, weight_style=:standard)
damis_dropout_data_DM = AMIS(N=Nsamps; d0=dropout_model(), P, Nsamps_per_update, update_distribution=train_model, sampler=default_sampler(f), Ntrials=10, α, weight_style=:DM)

plt = plot_convergence(mc_data, label="MC", title="Adaptive MIS estimate of VaR (10 trials)", xlabel="Samples", ylabel="VaR")
plot_convergence(amis_data_standard, label="deep AMIS - Standard Weights - var fit", p=plt)
plot_convergence(damis_dropout_data_standard, label="deep AMIS - dropout - standard Weights - var fit", p=plt)
# plot_convergence(damis_dropout_data_DM, label="deep AMIS - dropout Standard Weights - var fit", p=plt)

