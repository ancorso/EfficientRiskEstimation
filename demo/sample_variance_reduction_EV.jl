### A Pluto.jl notebook ###
# v0.18.0

using Markdown
using InteractiveUtils

# ╔═╡ 62758e9e-448c-430f-b0ed-280aeafe992e
begin
	using Pkg
	Pkg.add(path="/Users/anthonycorso/.julia/dev/ImportanceWeightedRiskMetrics")
	using ImportanceWeightedRiskMetrics
	using Distributions
	using Plots
	using Random
end

# ╔═╡ 78dda16c-a5a8-45f1-a434-2d3dc70d71af
f(x) = 1000*(min(2^x, 100.)) * (x > 4)

# ╔═╡ 3aeff4d9-bd93-4780-a971-b05434524041
P = Normal(0,1)

# ╔═╡ 23ea19dd-fea0-4325-abe6-83424e65e2c8
p(x) = pdf(P, x)

# ╔═╡ 7741efd0-d972-4906-8451-72b42a2eaf0e
MIS_dists = [Normal(0,1), Normal(3,.2), Normal(3.7,.2), Normal(4.4,.2), Normal(5.1,.2)]

# ╔═╡ 9b2d134d-38f7-43bd-a49c-b971f11a9a6e
begin
	xticks = -8:0.1:8
	plot(xticks, f, label="f =  2ˣ ⋅ 1{x > 4}", title="Integral to approximate", xlabel="x", ylims=(0,1), legend=:topleft)
	plot!(xticks, p, label="p(x) = N(x; 0,1)")
	for d in MIS_dists
		plot!(xticks, x->pdf(d,x), color=3, label="")
	end
	plot!([], label="MIS dists", color=3)
	plot!(xticks, (x)->f(x)*p(x), label="fp⋅100")
end

# ╔═╡ c8ddee0f-dab3-479e-9289-645e9f20ca80
function plot_approach(sampler; Nsamps=1000, Ntrials=10, p=plot(xscale=:log, xlims=(100,Inf)), label="", kwargs...)
	Nsamps = Int(Nsamps)
	estimates = Array{Float64}[]
	Nsamps_arrs = Array{Float64}[]
	for i=1:Ntrials
		Nsamps_arr, μs = sampler(Nsamps)
		push!(estimates, μs)
		push!(Nsamps_arrs, Nsamps_arr)
	end
	plot!(p, mean(Nsamps_arrs), mean(estimates), ribbon=std(estimates), label=label; kwargs...)
end

# ╔═╡ 1b1ae43f-3510-4b7e-8df3-57667354b79a
MC(Nsamps) = 1:Nsamps, cumsum(f.(rand(P, Nsamps))) ./ (1:Nsamps)

# ╔═╡ 0fe1eac7-72b2-49d8-86f4-3b206ed5fff7
function IS(dist)
	(Nsamps) -> begin
		xs = rand(dist, Nsamps)
		ws = pdf.(P, xs) ./ pdf.(dist, xs)
		1:Nsamps, cumsum(f.(xs) .* ws) ./ (1:Nsamps)
	end
end

# ╔═╡ 0804a826-5b5e-4ab0-8645-9c78cedf946e
function MIS(dists, weight_style = :standard) #weight styles [:standard, :DM]
	(Nsamps) -> begin
		xs = Float64[]
		ws = Float64[]
		while length(xs) < Nsamps
			for d in dists
				x = rand(d)
				if weight_style==:standard
					w = pdf(P, x) / pdf(d, x)
				elseif weight_style==:DM
					w = pdf(P, x) / mean([pdf(d, x) for d in dists])
				end
				push!(xs, x)
				push!(ws, w)
			end
		end
		1:Nsamps, cumsum(f.(xs) .* ws) ./ (1:Nsamps)
	end
end

# ╔═╡ 56004952-4162-47d3-b0d0-37866888c3f3
 function AMIS(weight_style = :standard; d0 = Normal(0,01)) #weight styles [:standard, :DM]
	(Nsamps; all_dists=[], xs = Float64[], ws = Float64[], ws_standard = Float64[], inter_ws=[]) -> begin
		dists = []
		all_weights = []
		ests = []
		Nsamps_arr = []
		d = deepcopy(d0)
		fs = Float64[]
		for i=1:Nsamps
			push!(all_dists, deepcopy(d))
			x = rand(d)
			push!(xs, x)
			push!(fs, f(x))

			push!(ws_standard, pdf(P, x) / pdf(d, x))
			if weight_style==:standard
				push!(ws, pdf(P, x) / pdf(d, x))
			elseif weight_style==:DM
				# apply previous pdfs to current sample
				# push!(all_weights, [pdf(P, x)])
				push!(all_weights, [pdf(dist, x) for dist in [P, dists...]])
				push!(ws, all_weights[end][1] / mean(all_weights[end][2:end]))
				
			end

			# if length(dists) > 100
			# 	index = rand(1:i)
			# 	if index <= 100
			# 		dists[index] = d
			# 	end
			# else 
			# push!(dists, deepcopy(d))
			# end

			
			

			if ((i) % 100) == 0
				if weight_style==:DM
					for j=1:length(xs) #rand(, min(i,2))
						push!(all_weights[j], pdf(d, xs[j]))
						ws[j] = all_weights[j][1] / mean(all_weights[j][2:end])
					end
				end
				
				push!(ests, sum(fs .* ws) / i)
				push!(inter_ws, deepcopy(ws))
				push!(Nsamps_arr, i)
				
				# Add the last distribution to the history
				weight_style==:DM && push!(dists, deepcopy(d))

				
				println("refitting at i=$i")

				
				# dnew = fit(Normal, xs, (ws .+ 1e-6) .* (fs .+ 1e-6))
				# Δμ = clamp(dnew.μ - d.μ, -0.1, 0.1)
				# Δσ = clamp(dnew.σ - d.σ, -0.05, 0.05)
				# d = Normal(d.μ + Δμ, d.σ + Δσ)
				d = fit(Normal, xs, (ws .+ 1e-6) .* (fs .+ 1e-6))

				# When we refit, add the new weight to the list for past samples

				
			end
		end
		# if weight_style==:standard
		# 	ests = cumsum(fs .* ws) ./ (1:Nsamps)
		# end
		
		Nsamps_arr, ests
	end
 end

# ╔═╡ 246fb1cd-bf2d-44f9-9a40-2c60ce0ee748
begin 
	Nsamps = 2e4
	plt = plot_approach(MC, Nsamps=1e5, label="MC", ylims=(0,2))

	## Importance sampling
	# plot_approach(IS(Normal(4, 0.2)), Nsamps=Nsamps, label="IS w/ N(3,.1)", p=plt)
	# plot_approach(IS(Normal(3, 1)), Nsamps=1e4, label="IS w/ N(3,1)", p=plt)
	# plot_approach(IS(TruncatedNormal(0, 1, 4, Inf)), Nsamps=Nsamps, label="IS w/ TN(0,1, 3, Inf)", p=plt)

	## Multiple importance sampling
	# plot_approach(MIS(MIS_dists), Nsamps=Nsamps, label="MIS - standard weights", p=plt)
	# plot_approach(MIS(MIS_dists, :DM), Nsamps=Nsamps, label="MIS - dm weights", p=plt)
	# plot_approach(MIS([P, MIS_dists[1], MIS_dists[end], MIS_dists[2]], :DM), Nsamps=Nsamps, label="defensive IS", p=plt)

	##Random adaptive MIS
	plot_approach(AMIS(d0=Normal(6,.2)), Nsamps=Nsamps, label="MIS - standard weights", p=plt, )
	plot_approach(AMIS(:DM, d0=Normal(6,.2)), Nsamps=Nsamps, label="MIS - dm weights", p=plt)
end

# ╔═╡ ba388d5e-4410-4439-9293-726cfade3c89
begin
	d0 = Normal(6,.2)
	Nsamp = 2500

	all_dists = []
	xs = Float64[]
	ws_dm = Float64[]
	ws_standard = Float64[]
	inter_ws = []
	ests_dm = AMIS(:DM, d0=deepcopy(d0))(Nsamp;all_dists, xs=xs, ws=ws_dm, ws_standard, inter_ws=inter_ws)
	fs = f.(xs)

	@info length(inter_ws)

	ws = ws_standard
	ests = cumsum(fs .* ws) ./ (1:Nsamp)

	
	pest = plot(ests, title="estimate", label="std", legend=:topleft)
	plot!(ests_dm, title="estimate", label="DM")

	hline!([sum(fs[end-1000:end] .* ws[end-1000:end]) / 1000])

	px = histogram(xs, title="xs")

	perm = sortperm(fs)
	pf = plot(fs[perm], ws[perm], title="fs vs weight", label="std", yscale=:log10)
	plot!(fs[perm], ws_dm[perm], label="DM")

	pw = plot(ws, title="weights", yscale=:log10, legend=:bottomright)
	plot!(ws_dm, label="DM")
	
	pdist = plot([d.μ for d in all_dists], label="mean - std", legend=:right)
	plot!([d.σ for d in all_dists], label="std - std")

	plot(pest, px, pf, pw, pdist, size=(1500,1000))
end


# ╔═╡ 66791c1f-1571-45c3-bc1e-2439e590cd13
begin
	loc = 5
	v = loc*100
	fperm = sortperm(fs[1:v])
	fperm = fperm[fs[fperm] .> 0]
	sortf = fs[fperm]


	p2 = plot(sortf, ws[fperm], label="std", marker=true)
	plot!(sortf, inter_ws[loc][fperm], yscale=:log10, label="Dm", xscale=:log10, xticks=10, marker=true)

	distmeans = [d.μ for d in all_dists[fperm]]
	distsigmas = [d.σ for d in all_dists[fperm]]
	@info string("indices: ", fperm[1:4], " means: ", distmeans[1:4], "sigmas: ", distsigmas[1:4], " xs: ", xs[fperm][1:4], " curr dist: ", all_dists[v])
	# plot(distmeans[1:4])



	

	# @info string("dm: ", sum(fs[fperm] .* inter_ws[loc][fperm]) / v, " std:", sum(fs[fperm] .* ws[fperm] / v))

	p2
end

# ╔═╡ 11ea763e-add2-4732-b40c-d3a2818f4d48


# ╔═╡ Cell order:
# ╠═62758e9e-448c-430f-b0ed-280aeafe992e
# ╠═78dda16c-a5a8-45f1-a434-2d3dc70d71af
# ╠═3aeff4d9-bd93-4780-a971-b05434524041
# ╠═23ea19dd-fea0-4325-abe6-83424e65e2c8
# ╠═7741efd0-d972-4906-8451-72b42a2eaf0e
# ╠═9b2d134d-38f7-43bd-a49c-b971f11a9a6e
# ╠═c8ddee0f-dab3-479e-9289-645e9f20ca80
# ╠═1b1ae43f-3510-4b7e-8df3-57667354b79a
# ╠═0fe1eac7-72b2-49d8-86f4-3b206ed5fff7
# ╠═0804a826-5b5e-4ab0-8645-9c78cedf946e
# ╠═56004952-4162-47d3-b0d0-37866888c3f3
# ╠═246fb1cd-bf2d-44f9-9a40-2c60ce0ee748
# ╠═ba388d5e-4410-4439-9293-726cfade3c89
# ╠═66791c1f-1571-45c3-bc1e-2439e590cd13
# ╠═11ea763e-add2-4732-b40c-d3a2818f4d48
