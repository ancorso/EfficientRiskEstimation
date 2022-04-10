function Distributions.logpdf(d::Dict{Symbol, Tuple{Sampleable, Int64}}, x, i)
    sum([logpdf(d[k][1], x[k][i]) for k in keys(d)])
end

function Distributions.logpdf(d::Dict{Symbol, Tuple{Sampleable, Int64}}, x)
    sum([logpdf(d, x, i) for i=1:length(first(x)[2])])
end

function Base.rand(d::Dict{Symbol, Tuple{Sampleable, Int64}})
    Dict(k => rand(d[k][1], d[k][2]) for k in keys(d))
end

function Base.rand(d::Dict{Symbol, Tuple{Sampleable, Int64}}, N::Int)
    [rand(d) for i=1:N]
end

function Distributions.fit(d::Dict{Symbol, Tuple{Sampleable, Int64}}, samples, weights; add_entropy = (x)->x)
    N = length(samples)
    new_d = Dict{Symbol, Tuple{Sampleable, Int64}}()
    for s in keys(d)
        dtype = typeof(d[s][1])
        m = d[s][2]
        all_samples = vcat([samples[j][s][:] for j=1:N]...)
        all_weights = vcat([fill(weights[j], length(samples[j][s][:])) for j=1:N]...)
        new_d[s] = (add_entropy(fit(dtype, all_samples, all_weights)), m)
    end
    new_d
end

# This version uses a vector of distributions for sampling
# N is the number of samples taken
# m is the length of the vector

# if batched is set to true, loss function must return an array containing loss values for each sample   
function cem(cost_weight_fn, d_in; α, min_samples_above=0.1, max_iter, N=100)
    d_var, d_cvar = deepcopy(d_in), deepcopy(d_in)
    
    all_costs = Float64[]
    all_weights = Float64[]
    
    for iteration in 1:max_iter
        # Get samples -> Nxm
        samples_var = rand(d_var, Int(N/2))
        samples_cvar =rand(d_cvar, Int(N/2))
        samples = [samples_var..., samples_cvar...]

        # sort the samples by loss and select elite number
        vals_var = [cost_weight_fn(d_var, s) for s in samples_var]
        vals_cvar = [cost_weight_fn(d_cvar, s) for s in samples_cvar]
        vals = [vals_var..., vals_cvar...]
        costs = [c for (c,_) in vals]
        weights = [w for (_,w) in vals]
        
        push!(all_costs, costs...)
        push!(all_weights, weights...)
        
        m = IWRiskMetrics(all_costs, all_weights, α)
        mfallback = IWRiskMetrics(all_costs, ones(length(all_costs)), min_samples_above)
        mfallback2 = IWRiskMetrics(costs, ones(length(costs)), min_samples_above)
        
        var_est = min(min(m.var, mfallback.var), mfallback2.var)

        println("iteration ", iteration, " of ", max_iter, " mean loss: ", mean(costs), " mean weight: ", mean(weights), " var: ", m.var, " cvar: ", m.cvar, " target var: ", var_est, "samples larger than var: ", sum((costs .> var_est)))
        
        weights_var = weights .* (costs .> var_est)
        weights_cvar = weights_var .* costs 

        d_var = fit(d_var, samples, weights_var)
        d_cvar = fit(d_cvar, samples, weights_cvar)
    end
    all_costs, all_weights, (d_var, d_cvar)
end

