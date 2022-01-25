using Plots
using ImportanceWeightedRiskMetrics


function get_samples(buffer::ExperienceBuffer, px)
    eps = episodes(buffer)
    vals = Float64[]
    weights = Float64[]
    for ep in eps
        eprange = ep[1]:ep[2]
        push!(vals, sum(buffer[:r][1,eprange]))
        nom_logprob = logpdf(px, buffer[:s][:,eprange], buffer[:a][:,eprange])
        push!(weights, exp(sum(nom_logprob .- buffer[:logprob][:,eprange])))
    end
    vals, weights
end


function compute_risk_cb(period, min_samples_above = 0.1)
    (𝒮; info=info) -> begin
        if ((𝒮.i + 𝒮.ΔN) % period) == 0
            α = 𝒮.𝒫[:α]
            px = 𝒮.𝒫[:px]
            vals, weights = get_samples(𝒮.buffer, px)
            m = IWRiskMetrics(vals, weights, α)
            mfallback = IWRiskMetrics(vals, ones(length(vals)), min_samples_above)
            
            𝒮.𝒫[:rα][1] = min(m.var, mfallback.var)
            
            # Log the metrics
            info["var"] = m.var
            info["cvar"] = m.cvar
            info["mean"] = m.mean
            info["worst"] = m.worst
            info["target_var"] = 𝒮.𝒫[:rα][1]
            
            # Update the running estimate of var
            
        end
    end
end


function running_risk_metrics(Z, w, α, Nsteps=10)
    imax = log10(length(Z))
    Δi = (imax-1)/Nsteps
    irange = 1:Δi:imax
    println(irange)
    rms = [IWRiskMetrics(Z[1:Int(floor(10^i))], w[1:Int(floor(10^i))], α) for i=irange]
    
    10 .^ irange, rms
end


function make_plots(Zs, ws, names, α, Nsteps=10)
    mean_plot=plot(title="Mean", legend=:bottomright, xscale=:log10)
    var_plot=plot(title="VaR", legend=:bottomright)
    cvar_plot=plot(title="CVaR", legend=:bottomright)
    worst_case=plot(title="Worst_case", legend=:bottomright)
    
    for (Z,w,name) in zip(Zs, ws, names)
        irange, rms = running_risk_metrics(Z, w, α, Nsteps)
        plot!(mean_plot, irange, [rm.mean for rm in rms], label=name)
        plot!(var_plot, irange, [rm.var for rm in rms], label=name)
        plot!(cvar_plot, irange, [rm.cvar for rm in rms], label=name)
        plot!(worst_case, irange, [rm.worst for rm in rms], label=name)
    end
    plot(mean_plot, var_plot, cvar_plot, worst_case, layout=(2,2))
end


function log_err_pf(π, D, ys)
    N = length(ys)
    sum([abs.(log.(value(n, D[:s], D[:a]) .+ eps())  .-  log.(y  .+ eps())) for (n, y) in zip(π.networks[1:N], ys)])    
end


function abs_err_pf(π, D, ys)
    N = length(ys)
    sum([abs.(value(n, D[:s], D[:a])  .-  y) for (n, y) in zip(π.networks[1:N], ys)])
end


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
