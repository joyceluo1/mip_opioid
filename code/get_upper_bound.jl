#= 

Obtains an upper bound on the optimal objective value and propagates the compartmental 
model values for a feasible solution to the treatment facility 
location and budget allocation optimization problem for set of US states.

=#

using LinearAlgebra, CSV, DataFrames, Plots, DelimitedFiles, Tables, MathOptInterface

states = ["AL", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI",
        "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN",
        "MO", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "OH", "OK",
       "OR", "PA", "RI", "SC", "TN", "TX", "UT", "VT", "VA", "WA",
       "WI"]

global obj_vals = Array{Float64}(undef, 1, 0)
global final_sols = Array{Float64}(undef, 3, 0)
for state in states
    # Loading data, feasible solution, and parameters
    param = DataFrame(CSV.File("avg_weighting_state_params.csv")) # resulting params from indiv_state_train_weighted.jl
    model_data = DataFrame(CSV.File("opt_model_data/pr_svi_" * state * "_allnorm.csv"))
    budget = DataFrame(CSV.File("state_samhsa_grant.csv"))
    model_msrs = DataFrame(CSV.File("rev2_orig_relaxation/model_msrs_new.csv"))
    opt_xs = DataFrame(CSV.File("rev2_orig_relaxation/" * state * "_x_dist.csv"))
    opt_xs = sort!(opt_xs, [:FIPS])
    opt_buds = DataFrame(CSV.File("rev2_orig_relaxation/" * state * "_bud_dist.csv"))
    opt_buds = sort!(opt_buds, [:FIPS])
    len = nrow(model_data)
    N = round(Int32, budget[in([state]).(budget.State), "add"][1] + sum(model_data[!, "n"]))
    data = readdlm("tuples/yearly_tuples_" * state * ".csv", ',', Float64)

    # Set hyperparameters
    if data[21,6] < 10000
        scale = 100
        inf = 10
        lambda_D = 10
        lambda_A = 9
    else
        scale = 1000
        inf = 9
        lambda_D = 10
        lambda_A = 9
    end
    lambda_pr = 0.001
    lambda_svi = 0.009
    lambda_pop = 0.001

    T = 9
    alpha = 0.15
    gamma = 0.00744
    delta = 0.1
    sigma = 0.9
    mu = param[in([state]).(param.State), "mu"][1]
    phi = param[in([state]).(param.State), "phi"][1]
    if phi < 1e-7
        phi = 0 
    end 
    epsilon = param[in([state]).(param.State), "epsilon"][1]
    beta = param[in([state]).(param.State), "beta"][1]
    if beta < 1e-7
        beta = 0
    end
    zeta = param[in([state]).(param.State), "zeta"][1]
    d = 448

    counties = DataFrame(CSV.File("county_pops.csv"))
    county_pops = counties[in([state]).(counties.STNAME), "POPESTIMATE2017"][2:end]
    pr = model_data[!, "pres_r_norm"]
    SVI = model_data[!, "RPL_THEMES"]
    pr_weight = []
    SVI_weight = []
    pop_weight = []
    for i in 1:length(pr)
        push!(pr_weight, 1/pr[i])
        push!(SVI_weight, 1/SVI[i])
        push!(pop_weight, (sum(county_pops)/county_pops[i]))
    end

    n = model_data[!, "n"]
    x = opt_xs[!, "x"]
    h = model_msrs[3, state]

    Δt = 0.25

    pops = DataFrame(CSV.File("pops.csv"))
    pop_bound = pops[21, state]
    pop = pops[19, state]/scale
    p1 = data[19,2]/scale
    i1 = data[19,3]/scale
    a1 = data[19,4]/scale
    r1 = data[19,5]/scale
    d1 = data[19,6]/scale
    s1 = pop - p1 - i1 - a1 - r1 - d1

    S = Array{Float64}(undef, T)
    P = Array{Float64}(undef, T)
    I = Array{Float64}(undef, T)
    A = Array{Float64}(undef, T)
    R = Array{Float64}(undef, T)
    D = Array{Float64}(undef, T)
    S[1] = s1
    P[1] = p1
    I[1] = i1
    A[1] = a1
    R[1] = r1
    D[1] = d1

    expr = []
    for t in 1:T
        push!(expr, sum((x[i]-n[i])*(opt_buds[!, string(t)][i]/(d*x[i])) for i in 1:len))
    end 

    # Propagating dynamics
    for j in 2:T
        i = j - 1 
        S[j] = (S[i]*scale + (-alpha*S[i]*scale+ epsilon*P[i]*scale + delta*R[i]*scale)*Δt)/scale
        P[j] = (P[i]*scale + (alpha*S[i]*scale - (epsilon + gamma + beta)*P[i]*scale)*Δt)/scale
        I[j] = (I[i]*scale + (beta*P[i]*scale - phi*I[i]*scale)*Δt)/scale
        A[j] = (A[i]*scale + (gamma*P[i]*scale + sigma*R[i]*scale + phi*I[i]*scale - zeta*A[i]*scale - expr[i] - mu*A[i]*scale)*Δt)/scale
        R[j] = (R[i]*scale + (zeta*A[i]*scale + expr[i] - (delta + sigma)*R[i]*scale)*Δt)/scale
        D[j] = (D[i]*scale + (mu*A[i]*scale)*Δt)/scale
    end
    # println(D)
    # println(A)

    # Calculating the objective value
    obj_val = lambda_D*D[T] + lambda_A*sum(A) + lambda_pr*sum(pr_weight.*x) + lambda_svi*sum(SVI_weight.*x) + inf*h + 
        lambda_pop*sum(sum(pop_weight.*(opt_buds[!, string(t)]./100000)) for t in 1:T)

    global obj_vals = hcat(obj_vals, obj_val)
    global final_sols = hcat(final_sols, [last(A)*scale, last(R)*scale, last(D)*scale])
end
CSV.write("rev2_orig_relaxation/obj_UBs.csv", Tables.table(obj_vals), writeheader = true, header = states)
CSV.write("rev2_orig_relaxation/final_UB_sols.csv", Tables.table(final_sols), writeheader = true, header = states)