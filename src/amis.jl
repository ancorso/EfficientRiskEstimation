using Distributions
using Random
using ImportanceWeightedRiskMetrics

## Multiple importance sampling distributions
mutable struct MISDistribution
	distributions
	Nsamps
	weight_style
	i
	MISDistribution(distributions, Nsamps=ones(length(distributions)); weight_style=:DM, i=0) = new(distributions, Nsamps, weight_style, i)
end

current_distribution(d::MISDistribution) = d.distributions[findfirst(cumsum(d.Nsamps) .>= d.i)]

# Define the pdf according to the DM weight scheme
function Distributions.pdf(d::MISDistribution, x)
	if d.weight_style == :standard
		return pdf(current_distribution(d), x)
	elseif d.weight_style == :DM
		return mean(N*pdf(dist, x) for (dist, N) in zip(d.distributions, d.Nsamps)) / mean(d.Nsamps)
	else 
		@error "Unrecognized weight style: $(d.weight_style)"
	end
end

# Define the sampler
function Base.rand(rng::AbstractRNG, d::MISDistribution)
	d.i = mod1(d.i+1, sum(d.Nsamps))
	rand(rng, current_distribution(d))
end

## MDP Samplers
struct MDPSampler
	sampler
end
MDPSampler(mdp, π) = MDPSampler(Sampler(mdp, π, required_columns=[:logprob]))

# Sampling
function Base.rand(rng::AbstractRNG, d::MDPSampler)
	D = episodes!(d.sampler, explore=true)
	D[:r][end], D
end

# Getting the pdf
Distributions.pdf(d::MDPSampler, x) = exp(sum(logpdf(d.sampler.agent.π, x[:s], x[:a])))

# default sampler
default_sampler(f) = (d)->begin
	x = rand(d)
	f(x), x
end

## AMIS
function AMIS(;N, # Total number of samples to get
			   P, # Nominal distribution
			   α=0.001, # value for VaR and CVaR
			   d0 = P, # Initial sampling distribution
			   update_distribution = (;d, kwargs...) -> d, # Function used to update the distribution
			   sampler = rand, # Function used to return a sample f and x
			   log_period = 100, # How frequently to compute the statistics
			   log_period_multiplier = 10^(1/3), # How much the logging increases each time
			   Nsamps_per_update=Inf, # Number of samples to take before updating the distribution
			   weight_style=:standard, # weight styles [:standard, :DM]
			   weight_fn = (d, P, x) -> pdf(P, x) / pdf(d, x), # Function to compute standard weights
			   Ntrials=1, # Number of trials to run
			   saveall = true
			  ) 
	datas = []
	for trial in 1:Ntrials
		dists, all_weights, data = [], [], Dict{Symbol, Any}(:var => Float64[], :Nsamps=>Int[], :cvar=>Float64[], :mean=>Float64[], :worst=>Float64[])
		data[:α] = α
		data[:d0] = deepcopy(d0)
		
		d = deepcopy(d0)
		xs, fs, ws = Any[], Float64[], Float64[]
		next_log = log_period
		
		for i=1:Int(N)
			# Generate and store the sample
			f, x = sampler(d)
			push!(xs, x)
			push!(fs, f)

			# Compute the impotance weight
			if weight_style==:standard
				push!(ws, weight_fn(d, P, x))
			elseif weight_style==:DM
				push!(all_weights, [pdf(dist, x) for dist in [P, dists..., d]])
				push!(ws, all_weights[end][1] / mean(all_weights[end][2:end]))
			end

			
			
			# Log the info
			if (i % next_log) == 0 || i==N
				
				next_log = floor(Int, next_log * log_period_multiplier)
				rm = IWRiskMetrics(fs, ws, α)
				push!(data[:Nsamps], i)
				push!(data[:var], rm.var)
				push!(data[:cvar], rm.cvar)
				push!(data[:mean], rm.mean)
				push!(data[:worst], rm.worst)
				println("logging at $i, var: $(rm.var)")
			end
			
			# Adapt the importance distribution
			if (i % Nsamps_per_update) == 0 
				d = update_distribution(;d, xs, fs, ws, α, all_weights, P)
				
				if weight_style==:DM
					for j=1:length(xs)
						push!(all_weights[j], pdf(d, xs[j]))
						ws[j] = all_weights[j][1] / mean(all_weights[j][2:end])
					end
				end
				
				push!(dists, deepcopy(d))
				
			end
		end
		if saveall
			data[:xs] = xs
			data[:dists] = [d0, dists...]
		end
		data[:fs] = fs
		data[:ws] = ws
		push!(datas, deepcopy(data))
	end
	Ntrials==1 ? datas[1] : datas
end

