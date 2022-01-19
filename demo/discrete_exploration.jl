mutable struct DiscreteExploration <: Policy
    probs
end

function Crux.exploration(π::DiscreteExploration, s; π_on, i)
    vals = value(π_on, s)
    probs = π.probs(s)
    
    ps = vals .* probs
    ps = ps ./ sum(ps)
    i = rand(Categorical(ps))
    if log(ps[i]) < -5.0
        println("ps: ", ps, " i: ", i)
    end
    [π_on.outputs[i]], log(ps[i])
end

