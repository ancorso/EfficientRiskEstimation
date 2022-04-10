using Plots

function save_data(dir, name, data; color=1, cdata=[], cnames=[], ccolors=[])
	BSON.@save "$dir/$(name)_data.bson" data
	
	pconv = plot(title="VaR Convergence", xlabel="Samples", ylabel="VaR", xscale=:log, xlims=(100,Inf))
	for (d, n, c) in zip(cdata, cnames, ccolors)
		plot_convergence(d, label=n, color=c, p=pconv)
	end
	
	pconv = plot_convergence(data, label=name, color=color, p=pconv)
	savefig("$dir/$(name)_convergence.png")
	
	pcdf = plot_cdf(data, label=name, xlabel="f", ylabel="1-cdf(f)")
	savefig("$dir/$(name)_cdfs.png")
	
	plot(pconv, pcdf, size=(1200, 400))
end


function plot_convergence(datas::T; key=:var, title=string(key), p=plot(xscale=:log, xlims=(100,Inf), title=title), label="", kwargs...) where {T<:Dict}
	plot!(p, d[:Nsamps], d[key], label=label; kwargs...)
end
	
function plot_convergence(datas::T; key=:var, title=string(key), p=plot(xscale=:log, xlims=(100,Inf), title=title), label="", kwargs...) where {T<:Array}
	results = [d[key] for d in datas]
	samps = [d[:Nsamps] for d in datas]
	@assert all(samps .== [samps[1]])
	
	plot!(p, samps[1], mean(results), ribbon=std(results), label=label; kwargs...)
end

function plot_cdf(data::T; label="", title="1 - CDF", p=hline([α], label="α=$(α)", title=title), kwargs...) where {T <: Array}
	for (d, i) in zip(data, 1:length(data))
		plot_cdf(d, label=string(label, "-$i"), p=p; kwargs...)
	end
	p
end
	
function plot_cdf(data::T; α=data[:α], label="", title="1 - CDF", p=hline([α], label="α=$(α)", title=title),  kwargs...) where {T <: Dict}
	perm = sortperm(data[:fs], rev=true)
	plot!(p, data[:fs][perm], cumsum(data[:ws][perm])./length(data[:ws]), yscale=:log10, label=string(label, " (VaR = $(data[:var][end]))"); kwargs...)
	vline!([data[:var][end]], label="", linestyle=:dash, color = p.series_list[end].plotattributes[:seriescolor])
end

function plot_pdfs(xticks, pdfs; color=:auto, labels=[1, length(pdfs)], p=plot(), xlabel="x", ylabel="pdf", title="Adaptive IS", label="", kwargs...)
	Ncolors = length(pdfs)
	plt = plot!(p, color_palette = palette(color, Ncolors); xlabel, ylabel, title, kwargs...)
	for (d, i) in zip(pdfs, 1:Ncolors)
		plot!(p, xticks, (x)->pdf(d, x), label=i in labels ? "$(label) - i=$i" : "")
	end
	plt
end

