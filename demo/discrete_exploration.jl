mutable struct DiscreteExploration <: Policy
    probs
end

function Crux.exploration(π::DiscreteExploration, s; π_on, i)
    if π_on isa MixtureNetwork && π_on.networks[π_on.current_net] isa DistributionPolicy
        return exploration(π_on, s)
    end
    
    vals = value(π_on, s)
    probs = π.probs(s)
    
    ps = vals .* probs
    ps = ps ./ sum(ps)
    i = rand(Categorical(ps))
    outputs = π_on isa MixtureNetwork ? π_on.networks[π_on.current_net].outputs : π_on.outputs
    [outputs[i]], log(ps[i])
end

