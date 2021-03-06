using Crux
using LinearAlgebra
using Distributions
using Random
using ImportanceWeightedRiskMetrics

# function value_estimate(ฯ::DiscreteNetwork, P, s)
# 	pdfs = Crux.logits(P, s)
# 	sum(value(ฯ, s) .* pdfs, dims=1)
# end

# function PVaR_target(ฯ, ๐, P, var_target)
# 	return ๐[:done] .* (๐[:r] .> var_target) .+ (1.f0 .- ๐[:done]) .* value_estimate(ฯ, P, ๐[:sp])
# end
# 
# function abs_err_pf(ฯ, ๐, ys)
# 	abs.(value(ฯ, ๐[:s], ๐[:a])  .-  ys)
# end

# Training
function trainer(๐ฎ; elite_frac=0.1, ฮฑscale = 1.1f0)
	function train_dist(;d, xs, fs, ws, ฮฑ, P, kwargs...)
		d = d isa MISDistribution ? d.distributions[end] : d #NOTE: Need to have the policy as the last distribution
		
		# Compute VaR Estimate
		risk_metric = IWRiskMetrics(fs, ws, ฮฑscale*ฮฑ)
		var_min = sort(fs, rev=true)[floor(Int, length(fs) * elite_frac)]
		๐ฎ.๐ซ[:var_target][1] = min(risk_metric.var, var_min)
		var_info = Dict(:curr_var_est => risk_metric.var, :elite_estimate => var_min)
		
		# Construct the experience buffer
		๐ = Dict(k => hcat([d[k] for d in xs]...) for k in keys(xs[1]))
		๐[:return] = hcat([fill(Float32(f), 1, length(x[:r])) for (f, x) in zip(fs, xs)]...)
		๐[:cum_importance_weight] = hcat([fill(Float32(w), 1, length(x[:r])) for (w, x) in zip(ws, xs)]...)
		๐ = ExperienceBuffer(๐)
		
		# Train the actor
		info = Dict()
		Crux.batch_train!(actor(d.sampler.agent.ฯ), ๐ฎ.a_opt, ๐ฎ.๐ซ, ๐, info=info, ฯ_loss=d.sampler.agent.ฯ)
		
		# Train the critic (if applicable)
		if !isnothing(๐ฎ.c_opt)
			Crux.batch_train!(critic(d.sampler.agent.ฯ), ๐ฎ.c_opt, ๐ฎ.๐ซ, ๐, info=info)
		end
		
		# Log the results
		๐ฎ.log.sampler = d.sampler
        Crux.log(๐ฎ.log, ๐ฎ.i + 1:๐ฎ.i + ๐ฎ.ฮN, info, var_info, ๐ฎ=๐ฎ)
    	๐ฎ.i += ๐ฎ.ฮN
		
		d
	end
end

# Loss function
function var_loss(ฯ, ๐ซ, ๐; info = Dict())
	new_probs = logpdf(ฯ, ๐[:s], ๐[:a])
	# nom_probs = logpdf(P.ฯ, ๐[:s], ๐[:a])
	# e_loss = -mean(entropy(ฯ, ๐[:s]))
	
	Flux.Zygote.ignore() do
		# info[:entropy] = -e_loss
		info[:kl] = mean(๐[:logprob] .- new_probs)
		info[:mean_weight] = mean(๐[:cum_importance_weight])
		info[:target] = ๐ซ[:var_target][1]
		
	end 
	
	-mean(new_probs .* ((๐[:return] .> ๐ซ[:var_target][1]) .* ๐[:cum_importance_weight] .- Float32(๐ซ[:ฮฑ]))) #+ 0.01f0*e_loss
	# -mean(new_probs .* ((๐[:return] .> ๐ซ[:var_target][1]) .* ๐[:cum_importance_weight] .- value(ฯ, ๐[:s]))) #+ 0.01f0*e_loss
	# -mean(new_probs .* (๐[:return] .> ๐ซ[:var_target][1]))
end

# function baseline_loss(ฯ, ๐ซ, D; kwargs...)
# 	Flux.mse(value(ฯ, D[:s]), ((D[:return] .> ๐ซ[:var_target][1]) .* D[:cum_importance_weight]))
# end

function policy_match_logits(P)
    logps = log.(Float32.(P.distribution.p))
    (ฯ, s) -> softmax(log.(softplus.(value(ฯ, s))) .+ logps)
end

function DeepSampler(;Px, mdp, ฯfn, ฮฑ, elite_frac=0.1, ฮฑscale=1.1f0, actor_batch_size=1024, target_kl=0.1f0)
    # Solver setup for training
    ๐ฎ = OnPolicySolver(;agent=PolicyParams(ฯ=ฯfn(), pa=Px),
                    S=state_space(mdp),
                    ๐ซ = (;var_target=[0.0],ฮฑ),
                    log = LoggerParams(;dir="log/DeepAMIS"),
                    a_opt = TrainingParams(;loss=var_loss, name = "actor_", batch_size=actor_batch_size, early_stopping = (infos) -> (infos[end][:kl] > target_kl)),
                    # c_opt=TrainingParams(;loss=baseline_loss, name = "critic_", max_batches=50, batch_size=1024),
                    required_columns = unique([:logprob, :importance_weight]))
                    
    sampler = Sampler(mdp, ๐ฎ.agent, S=๐ฎ.S, required_columns=๐ฎ.required_columns)
    ๐ฎ.log.sampler = sampler
    
    return (d0=MDPSampler(sampler), update_distribution=trainer(๐ฎ; elite_frac=elite_frac, ฮฑscale=ฮฑscale))
end



