using CSV
using DataFrames
using POMDPModelTools
using POMDPPolicies
using POMDPs

# Path to finance data in csv.
# E.g., https://github.com/AminHP/gym-anytrading/blob/master/gym_anytrading/datasets/data/STOCKS_GOOGL.csv.
path = "/Users/kykim/dev/gym-anytrading/gym_anytrading/datasets/data/STOCKS_GOOGL.csv"

df = CSV.read(path, DataFrame)

window_size = 10
frame_bound = [10, 300]

start_idx = frame_bound[1] - window_size + 1
end_idx = frame_bound[2]
prices = df[start_idx:end_idx, :Close]
features = convert(Matrix{Float64}, df[start_idx:end_idx, [:Close, :Open, :High, :Low]])

mdp = StockTrade(prices=prices, features=features)

function random_run(mdp::StockTrade)
    policy = FunctionPolicy((s) -> rand() > 0.5 ? :sell : :buy)
    s = rand(initialstate(mdp))
    r_total = 0.0
    while !isterminal(mdp, s)
        a = action(policy, s)
        r = reward(mdp, s, a)
        s = rand(transition(mdp, s, a))
        r_total += discount(mdp) * r
    end
    return r_total
end

println("Total r: ", random_run(mdp))
