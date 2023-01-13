#= 

Documentation: 
https://jump.dev/JuMP.jl/stable/
https://docs.juliahub.com/Gurobi/do9v6/0.7.7/

Treatment facility location and budget allocation optimization model for set of US states
using JuMP and Gurobi.

=#

using JuMP, Gurobi, LinearAlgebra, CSV, DataFrames, Plots, DelimitedFiles, ArgParse, Tables, MathOptInterface

states = ["AL", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI",
        "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN",
        "MO", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "OH", "OK",
       "OR", "PA", "RI", "SC", "TN", "TX", "UT", "VT", "VA", "WA",
       "WI"]
global cat = Array{Float64}(undef, 3, 0)
global msrs = Array{Float64}(undef, 2, 0)
global zeta_time = Array{Float64}(undef, 9, 0)
for state in states
    # Getting data and defining parameters
    param = DataFrame(CSV.File("state_params.csv")) # resulting params from indiv_state_train_updated.jl
    model_data = DataFrame(CSV.File("opt_model_data/pr_svi_" * state * ".csv"))
    budget = DataFrame(CSV.File("state_samhsa_grant.csv"))
    len = nrow(model_data)
    N = budget[in([state]).(budget.State), "add"][1] + sum(model_data[!, "n"])
    T = 9
    alpha = 0.15
    gamma = 0.00744
    delta = 0.1
    sigma = 0.9
    mu = param[in([state]).(param.State), "mu"][1]
    phi = param[in([state]).(param.State), "phi"][1]
    epsilon = param[in([state]).(param.State), "epsilon"][1]
    beta = param[in([state]).(param.State), "beta"][1]
    zeta = param[in([state]).(param.State), "zeta"][1]
    d = 448
    d_inv = d^-1
    d_lim = fill(budget[in([state]).(budget.State), "Quarterly_Budget"][1], T)

    counties= DataFrame(CSV.File("county_pops.csv"))
    county_pops = counties[in([state]).(counties.STNAME), "POPESTIMATE2017"][2:end]
    pr = model_data[!, "pres_r"]
    SVI = model_data[!, "RPL_THEMES"]
    n = model_data[!, "n"]
    # Setting hyperparameters
    lambda_A = 0.9
    lambda_pr = 0.3
    push = 0.1
    if state in ["MI", "WA", "FL"]
        inf = 475
    elseif state == "IL"
        inf = 500
    elseif state == "NJ"
        inf = 276
    elseif state == "MD"
        inf = 175
    elseif state == "MA"
        inf = 123
    else 
        inf = 450
    end
    Δt = 0.25
    data = readdlm("tuples/yearly_tuples_" * state * ".csv", ',', Float64)
    pops = DataFrame(CSV.File("pops.csv"))
    pop_bound = pops[21, state]

    model_nc = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model_nc, "NonConvex", 2)
    set_optimizer_attribute(model_nc, "TimeLimit", 2000)

    # Define variables
    @variables(model_nc, begin
            1 ≤ x[1:len] ≤ N, Int
            0 ≤ z[1:len] 
            0 ≤ S[1:T] 
            0 ≤ P[1:T] 
            0 ≤ I[1:T] 
            0 ≤ A[1:T] 
            0 ≤ R[1:T] 
            0 ≤ D[1:T] 
            0 ≤ d_bar[1:len, 1:T] 
            0 ≤ u[1:len]
            0 ≤ v[1:len]
            0 ≤ w[1:T]
            0 ≤ h
            end)


    # Fix initial conditions (starting in 2017/2018) 
    pop = pops[19, state]
    p1 = data[19,2]
    i1 = data[19,3]
    a1 = data[19,4]
    r1 = data[19,5]
    d1 = data[19,6]
    s1 = pop - p1 - i1 - a1 - r1 - d1
    fix(S[1], s1; force = true)
    fix(P[1], p1; force = true)
    fix(I[1], i1; force = true)
    fix(A[1], a1; force = true)
    fix(R[1], r1; force = true)
    fix(D[1], d1; force = true)

    # Define objective
    @objective(model_nc, Min, D[T] + lambda_A*sum(A) + lambda_pr*sum(u) + (1-lambda_pr)*sum(v) + push*sum(w) + inf*h)
    # Define constraints
    @constraint(model_nc, sum(x) - N ≤ h)
    @constraint(model_nc, [i = 1:len], x[i] >= n[i])
    @constraint(model_nc, [i = 1:len], x[i]*z[i] == 1)
    @constraint(model_nc, [t = 1:T], sum(d_bar[:, t]) <= d_lim[t])
    @expression(model_nc, expr[t = 1:T], sum(d_inv*(d_bar[i,t] - n[i]*z[i]*d_bar[i,t]) for i in 1:len))
    @constraint(model_nc, [i = 1:len], x[i] - (pr[i]/sum(pr))*N ≤ u[i])
    @constraint(model_nc, [i = 1:len], -(x[i] - (pr[i]/sum(pr))*N) ≤ u[i])
    @constraint(model_nc, [i = 1:len], x[i] - (SVI[i]/sum(SVI))*N ≤ v[i])
    @constraint(model_nc, [i = 1:len], -(x[i] - (SVI[i]/sum(SVI))*N) ≤ v[i])
    @constraint(model_nc, [i = 1:len, t = 1:T], d_bar[i, t] - (county_pops[i]/sum(county_pops))*d_lim[t] ≤ w[t])
    @constraint(model_nc, [i = 1:len, t = 1:T], -(d_bar[i, t] - (county_pops[i]/sum(county_pops))*d_lim[t]) ≤ w[t])
    # Epi model dynamics constraints
    for j in 2:T
        i = j - 1  
        @constraint(model_nc, S[j] >= S[i] + (-alpha*S[i] + epsilon*P[i] + delta*R[i])*Δt)
        @constraint(model_nc, P[j] >= P[i] + (alpha*S[i] - (epsilon + gamma + beta)*P[i])*Δt)
        @constraint(model_nc, I[j] >= I[i] + (beta*P[i] - phi*I[i])*Δt)
        @constraint(model_nc, A[j] >= A[i] + (gamma*P[i] + sigma*R[i] + phi*I[i] - zeta*A[i] - expr[i] - mu*A[i])*Δt)
        @constraint(model_nc, R[j] >= R[i] + (zeta*A[i] + expr[i] - (delta + sigma)*R[i])*Δt)
        @constraint(model_nc, D[j] >= D[i] + (mu*A[i])*Δt)
    end

    optimize!(model_nc)
    # Printing output
    println("Optimal objective: ", objective_value(model_nc))
    S_pred = []
    P_pred = []
    I_pred = []
    A_pred = []
    R_pred = []
    D_pred = []
    mod_zeta = []
    for i in 1:T
        push!(mod_zeta, zeta + (value(expr[i])/value(A[i])))
    end
    global zeta_time = hcat(zeta_time, mod_zeta)
    for i in 1:T
        push!(S_pred, value(S[i]))
    end
    for i in 1:T
        push!(P_pred, value(P[i]))
    end
    for i in 1:T
        push!(I_pred, value(I[i]))
    end
    for i in 1:T
        push!(A_pred, value(A[i]))
    end
    for i in 1:T
        push!(R_pred, value(R[i]))
    end
    for i in 1:T
        push!(D_pred, value(D[i]))
    end
    x_dist = Dict()
    for i in 1:len
        x_dist[model_data[i, "FIPS"]] = value(x[i])
        # println(model_data[i, "COUNTY"]  * ": " * string(value(x[i])))
    end
    CSV.write("new_updated_obj_model/" * state * "_x_dist.csv", x_dist, writeheader=true, header=["FIPS", "x"])
    # for i in 1:T
    #     println(i)
    #     for j in 1:len
    #         println(model_data[j, "COUNTY"]  * ": " * string(value(d_bar[j, i])))
    #     end
    # end
    bud_dist = Dict()
    for j in 1:len
        bud_dist[model_data[j, "FIPS"]] = round(value(d_bar[j, 1]), digits = 2)
    end
    CSV.write("new_updated_obj_model/" * state * "_bud_dist.csv", bud_dist,  writeheader=true, header=["FIPS", "bud"])
    vals = []
    push!(vals, MOI.get(model_nc, MOI.RelativeGap()))
    push!(vals, value(h))
    global msrs = hcat(msrs, vals)
    # println(S_pred)
    # println(P_pred)
    # println(I_pred)
    # println(A_pred)
    # println(R_pred)
    # println(D_pred)
    sols = []
    push!(sols, last(A_pred))
    push!(sols, last(R_pred))
    push!(sols, last(D_pred))
    global cat = hcat(cat, sols) 
    # pl = plot(0:T-1,P_pred, xlim = (0, 8), lw = 2, legend=:outertopright, label = "P")
    # plot!(pl, 0:T-1,I_pred, lw = 2, label = "I")
    # plot!(pl, 0:T-1,A_pred, lw = 2, label = "A")
    # plot!(pl, 0:T-1,R_pred, lw = 2, label = "R")
    # plot!(pl, 0:T-1,D_pred, lw = 2, label = "D")
    # ylabel!(pl,"Population")
    # xlabel!(pl,"Time")
    # display(pl)
end

CSV.write("new_updated_obj_model/final_sols.csv", Tables.table(cat), writeheader = true, header = states)
CSV.write("new_updated_obj_model/model_msrs_new.csv", Tables.table(msrs), writeheader = true, header = states)
CSV.write("new_updated_obj_model/zeta_over_time.csv", Tables.table(zeta_time), writeheader = true, header = states)