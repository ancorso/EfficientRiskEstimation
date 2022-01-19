using Plots

function get_samples(buffer::ExperienceBuffer, px)
    eps = episodes(buffer)
    vals = Float64[]
    weights = Float64[]
    for ep in eps
        eprange = ep[1]:ep[2]
        push!(vals, sum(buffer[:r][1,eprange]))
        nom_logprob = sum(logpdf(px, buffer[:s][:,eprange]) .* buffer[:a][:,eprange], dims=1)
        push!(weights, exp(sum(nom_logprob .- buffer[:logprob][:,eprange])))
    end
    vals, weights
end

function running_risk_metrics(Z, w, α, Nsteps=10)
    imax = log10(length(Z))
    Δi = (imax-1)/Nsteps
    irange = 1:Δi:imax
    println(irange)
    rms = [IWRiskMetrics(Z[1:Int(floor(10^i))], w[1:Int(floor(10^i))], α) for i=irange]
    
    irange, rms
end


function make_plots(Zs, ws, names, α, Nsteps=10)
    mean_plot=plot(title="Mean")
    var_plot=plot(title="VaR")
    cvar_plot=plot(title="CVaR")
    worst_case=plot(title="Worst_case")
    
    for (Z,w,name) in zip(Zs, ws, names)
        irange, rms = running_risk_metrics(Z, w, α, Nsteps)
        plot!(mean_plot, irange, [rm.mean for rm in rms], label=name)
        plot!(var_plot, irange, [rm.var for rm in rms], label=name)
        plot!(cvar_plot, irange, [rm.cvar for rm in rms], label=name)
        plot!(worst_case, irange, [rm.worst for rm in rms], label=name)
    end
    plot(mean_plot, var_plot, cvar_plot, worst_case, layout=(2,2))
end

