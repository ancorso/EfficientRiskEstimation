using Crux
using LinearAlgebra
using Distributions
using Random
using ImportanceWeightedRiskMetrics

# function value_estimate(π::DiscreteNetwork, P, s)
# 	pdfs = Crux.logits(P, s)
# 	sum(value(π, s) .* pdfs, dims=1)
# end

# function PVaR_target(π, 𝒟, P, var_target)
# 	return 𝒟[:done] .* (𝒟[:r] .> var_target) .+ (1.f0 .- 𝒟[:done]) .* value_estimate(π, P, 𝒟[:sp])
# end
# 
# function abs_err_pf(π, 𝒟, ys)
# 	abs.(value(π, 𝒟[:s], 𝒟[:a])  .-  ys)
# end

# Training
function trainer(𝒮; elite_frac=0.1, αscale = 1.1f0)
	function train_dist(;d, xs, fs, ws, α, P, kwargs...)
		d = d isa MISDistribution ? d.distributions[end] : d #NOTE: Need to have the policy as the last distribution
		
		# Compute VaR Estimate
		risk_metric = IWRiskMetrics(fs, ws, αscale*α)
		var_min = sort(fs, rev=true)[floor(Int, length(fs) * elite_frac)]
		𝒮.𝒫[:var_target][1] = min(risk_metric.var, var_min)
		var_info = Dict(:curr_var_est => risk_metric.var, :elite_estimate => var_min)
		
		# Construct the experience buffer
		𝒟 = Dict(k => hcat([d[k] for d in xs]...) for k in keys(xs[1]))
		𝒟[:return] = hcat([fill(Float32(f), 1, length(x[:r])) for (f, x) in zip(fs, xs)]...)
		𝒟[:cum_importance_weight] = hcat([fill(Float32(w), 1, length(x[:r])) for (w, x) in zip(ws, xs)]...)
		𝒟 = ExperienceBuffer(𝒟)
		
		# Train the actor
		info = Dict()
		Crux.batch_train!(actor(d.sampler.agent.π), 𝒮.a_opt, 𝒮.𝒫, 𝒟, info=info, π_loss=d.sampler.agent.π)
		
		# Train the critic (if applicable)
		if !isnothing(𝒮.c_opt)
			Crux.batch_train!(critic(d.sampler.agent.π), 𝒮.c_opt, 𝒮.𝒫, 𝒟, info=info)
		end
		
		# Log the results
		𝒮.log.sampler = d.sampler
        Crux.log(𝒮.log, 𝒮.i + 1:𝒮.i + 𝒮.ΔN, info, var_info, 𝒮=𝒮)
    	𝒮.i += 𝒮.ΔN
		
		d
	end
end

# Loss function
function var_loss(π, 𝒫, 𝒟; info = Dict())
	new_probs = logpdf(π, 𝒟[:s], 𝒟[:a])
	# nom_probs = logpdf(P.π, 𝒟[:s], 𝒟[:a])
	e_loss = -mean(entropy(π, 𝒟[:s]))
	
	Flux.Zygote.ignore() do
		info[:entropy] = -e_loss
		info[:kl] = mean(𝒟[:logprob] .- new_probs)
		info[:mean_weight] = mean(𝒟[:cum_importance_weight])
		info[:target] = 𝒫[:var_target][1]
		
	end 
	
	-mean(new_probs .* ((𝒟[:return] .> 𝒫[:var_target][1]) .* 𝒟[:cum_importance_weight] .- Float32(α))) #+ 0.01f0*e_loss
	# -mean(new_probs .* ((𝒟[:return] .> 𝒫[:var_target][1]) .* 𝒟[:cum_importance_weight] .- value(π, 𝒟[:s]))) #+ 0.01f0*e_loss
	# -mean(new_probs .* (𝒟[:return] .> 𝒫[:var_target][1]))
end

# function baseline_loss(π, 𝒫, D; kwargs...)
# 	Flux.mse(value(π, D[:s]), ((D[:return] .> 𝒫[:var_target][1]) .* D[:cum_importance_weight]))
# end

function policy_match_logits(P)
    logps = log.(Float32.(P.distribution.p))
    (π, s) -> softmax(log.(softplus.(value(π, s))) .+ logps)
end

function DeepSampler(;Px, mdp, πfn, elite_frac=0.1, αscale=1.1f0, actor_batch_size=1024, target_kl=0.1f0)
    # Solver setup for training
    𝒮 = OnPolicySolver(;agent=PolicyParams(π=πfn(), pa=Px),
                    S=state_space(mdp),
                    𝒫 = (;var_target=[0.0]),
                    log = LoggerParams(;dir="log/DeepAMIS"),
                    a_opt = TrainingParams(;loss=var_loss, name = "actor_", batch_size=actor_batch_size, early_stopping = (infos) -> (infos[end][:kl] > target_kl)),
                    # c_opt=TrainingParams(;loss=baseline_loss, name = "critic_", max_batches=50, batch_size=1024),
                    required_columns = unique([:logprob, :importance_weight]))
                    
    sampler = Sampler(mdp, 𝒮.agent, S=𝒮.S, required_columns=𝒮.required_columns)
    𝒮.log.sampler = sampler
    
    return (d0=MDPSampler(sampler), update_distribution=trainer(𝒮; elite_frac=elite_frac, αscale=αscale))
end



