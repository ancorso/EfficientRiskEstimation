"""Julia POMDP definition of the gym-anytrading env.

See https://github.com/AminHP/gym-anytrading.
"""

using Parameters
using POMDPs


@with_kw mutable struct StockTrade <: MDP{Matrix{Float64}, Symbol}
    # Options specified with keyword args.
    discount::Float64 = 1.0
    prices::AbstractVector{Float64}  # Vector of prices
    features::Matrix{Float64}  # 2D array of features
    window_size::Int = 10  # Size of the feature window.

    @assert size(prices)[1] == size(features)[1]

    # Internal constants and variables.
    _start_tick::Int = window_size
    _end_tick = length(prices)
    _current_tick::Int = _start_tick
    _last_trade_tick::Int = _current_tick - 1
    _position::Symbol = :short
end


# Utils.
function is_trade(action::Symbol, position::Symbol)
    return (action == :buy && position == :short) || (action == :sell && position == :long)
end


# States.
POMDPs.initialstate(mdp::StockTrade) = Deterministic(mdp.features[(mdp._start_tick - mdp.window_size + 1):mdp._start_tick, :])


# Actions.
POMDPs.actions(mdp::StockTrade) = (:sell, :buy)

const aind = Dict(:sell=>1, :buy=>2)
POMDPs.actionindex(mdp::StockTrade, a::Symbol) = aind[a]


# Transitions.
POMDPs.isterminal(m::StockTrade, s::Matrix{Float64}) = any(s .< 0.0)

function POMDPs.transition(mdp::StockTrade, s::Matrix{Float64}, a::Symbol)
    if mdp._current_tick == mdp._end_tick
        return Deterministic(fill(-1.0, size(s)))
    end

    if is_trade(a, mdp._position)
        # Reverse the position.
        mdp._position = mdp._position == :short ? :long : :short
        mdp._last_trade_tick = mdp._current_tick
    end

    observation = mdp.features[(mdp._current_tick - mdp.window_size + 1):mdp._current_tick, :]
    mdp._current_tick += 1
    return Deterministic(observation)
end


# Rewards.
function POMDPs.reward(mdp::StockTrade, s::Matrix{Float64}, a::Symbol)
    position = mdp._position
    if !is_trade(a, position) || position == :short
        return 0.0
    end

    current_price = mdp.prices[mdp._current_tick]
    last_trade_price = mdp.prices[mdp._last_trade_tick]
    return current_price - last_trade_price
end


# Discount.
POMDPs.discount(mdp::StockTrade) = mdp.discount
