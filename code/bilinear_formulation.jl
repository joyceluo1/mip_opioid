#= 

Documentation: 
https://jump.dev/JuMP.jl/stable/
https://docs.juliahub.com/Gurobi/do9v6/0.7.7/

Normalized treatment facility location and budget allocation optimization model for set of US states
using JuMP and Gurobi.

=#

using JuMP, Gurobi, LinearAlgebra, CSV, DataFrames, Plots, DelimitedFiles, ArgParse, Tables, MathOptInterface

states = ["AL", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI",
        "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN",
        "MO", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "OH", "OK",
       "OR", "PA", "RI", "SC", "TN", "TX", "UT", "VT", "VA", "WA",
       "WI"]

global cat = Array{Float64}(undef, 3, 0)
global msrs = Array{Float64}(undef, 4, 0)
# global zeta_time = Array{Float64}(undef, 9, 0)
for state in states
    # Getting data and defining parameters
    param = DataFrame(CSV.File("avg_weighting_state_params.csv")) # resulting params from indiv_state_train_weighted.jl
    model_data = DataFrame(CSV.File("opt_model_data/pr_svi_" * state * "_allnorm.csv"))
    budget = DataFrame(CSV.File("state_samhsa_grant.csv"))
    len = nrow(model_data)
    add = budget[in([state]).(budget.State), "add"][1]
    N = round(Int32, add + sum(model_data[!, "n"]))
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
    d_inv = d^-1
    d_lim = fill(budget[in([state]).(budget.State), "Quarterly_Budget"][1], T)

    counties= DataFrame(CSV.File("county_pops.csv"))
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
    println(pop_weight)
    println(pr_weight)
    n = model_data[!, "n"]

    Δt = 0.25
    pops = DataFrame(CSV.File("pops.csv"))
    pop_bound = pops[21, state]

    model_nc = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model_nc, "NonConvex", 2)
    set_optimizer_attribute(model_nc, "TimeLimit", 2000)
    # set_optimizer_attribute(model_nc, "NumericFocus", 3)

    # Define variables
    @variables(model_nc, begin
            1 <= x[1:len] <= N, Int
            z[1:len]
            0 ≤ S[1:T] 
            0 ≤ P[1:T] 
            0 ≤ I[1:T] 
            0 ≤ A[1:T] 
            0 ≤ R[1:T] 
            0 ≤ D[1:T] 
            d_bar[1:len, 1:T]
            0 ≤ h
            end)

    # Fix initial conditions (starting in 2017/2018) 
    pop = pops[19, state]/scale
    p1 = data[19,2]/scale
    i1 = data[19,3]/scale
    a1 = data[19,4]/scale
    r1 = data[19,5]/scale
    d1 = data[19,6]/scale
    s1 = pop - p1 - i1 - a1 - r1 - d1
    fix(S[1], s1; force = true)
    fix(P[1], p1; force = true)
    fix(I[1], i1; force = true)
    fix(A[1], a1; force = true)
    fix(R[1], r1; force = true)
    fix(D[1], d1; force = true)

    # UBs and LBs
    ubs = zeros(len, T)
    lbs = zeros(len, T)
    for i in 1:len
        for t in 1:T
            if state == "DE"
                ubs[i, t] = 14000*(n[i] + add)
            elseif state == "HI"
                ubs[i, t] = 32000*(n[i] + add)
            elseif state == "NV"
                ubs[i, t] = 12000*(n[i] + add)
            elseif state == "RI"
                ubs[i, t] = 5000*(n[i] + add)
            else
                ubs[i, t] = 10000*(n[i] + add)
            end
            if n[i] == 0.0
                lbs[i, t] = 5000
            else
                lbs[i, t] = 5000*n[i]
            end
        end
    end
    println(ubs[:, 1])

    # Define objective
    @objective(model_nc, Min, lambda_D*D[T] + lambda_A*sum(A) + lambda_pr*sum(pr_weight.*x) + 
        lambda_svi*sum(SVI_weight.*x) + inf*h + lambda_pop*sum(sum(pop_weight.*d_bar[:, t]) for t in 1:T))
    # Define constraints
    @constraint(model_nc, sum(x) - N ≤ h)
    @constraint(model_nc, [i = 1:len], x[i] >= n[i])

    @constraint(model_nc, [t = 2:T], d_bar[:, t-1] - d_bar[:, t] .<= 0.1)

    @constraint(model_nc, [i = 1:len], x[i]*z[i] == 1)

    @constraint(model_nc, [t = 1:T], sum(d_bar[:, t]*100000) <= d_lim[t])
    @constraint(model_nc, [i = 1:len, t = 1:T], lbs[i, t]/100000 <= d_bar[i, t] <= ubs[i, t]/100000)
    @expression(model_nc, expr[t = 1:T], sum(d_inv*(d_bar[i,t]*100000 - n[i]*d_bar[i,t]*z[i]*100000) for i in 1:len))

    @constraint(model_nc, [i = 1:len, t = 1:T], d_bar[i,t]*z[i] >= 0.05)
    # Epi model dynamics constraints
    for j in 2:T
        i = j - 1  
        @constraint(model_nc, S[j]*scale == S[i]*scale + (-alpha*S[i]*scale + epsilon*P[i]*scale + delta*R[i]*scale)*Δt)
        @constraint(model_nc, P[j]*scale == P[i]*scale + (alpha*S[i]*scale - (epsilon + gamma + beta)*P[i]*scale)*Δt)
        @constraint(model_nc, I[j]*scale == I[i]*scale + (beta*P[i]*scale - phi*I[i]*scale)*Δt)
        @constraint(model_nc, A[j]*scale == A[i]*scale + (gamma*P[i]*scale + sigma*R[i]*scale + phi*I[i]*scale - zeta*A[i]*scale - expr[i] - mu*A[i]*scale)*Δt)
        @constraint(model_nc, R[j]*scale == R[i]*scale + (zeta*A[i]*scale + expr[i] - (delta + sigma)*R[i]*scale)*Δt)
        @constraint(model_nc, D[j]*scale == D[i]*scale + (mu*A[i]*scale)*Δt)
    end

    optimize!(model_nc)
    # Printing output
    println("Optimal objective: ", objective_value(model_nc))
    print("Solve Time: ", solve_time(model_nc))
    S_pred = []
    P_pred = []
    I_pred = []
    A_pred = []
    R_pred = []
    D_pred = []
    mod_zeta = []

    # for i in 1:T
    #     push!(mod_zeta, zeta + (value(expr[i])/(value(A[i])*scale)))
    # end
    # global zeta_time = hcat(zeta_time, mod_zeta)
    for i in 1:T
        push!(S_pred, value(S[i])*scale)
    end
    for i in 1:T
        push!(P_pred, value(P[i])*scale)
    end
    for i in 1:T
        push!(I_pred, value(I[i])*scale)
    end
    for i in 1:T
        push!(A_pred, value(A[i])*scale)
    end
    for i in 1:T
        push!(R_pred, value(R[i])*scale)
    end
    for i in 1:T
        push!(D_pred, value(D[i])*scale)
    end
    x_dist = OrderedDict{Int, Float32}()
    for i in 1:len
        x_dist[model_data[i, "FIPS"]] = value(x[i])
        # println(model_data[i, "COUNTY"]  * ": " * string(value(x[i])))
    end
    println(x_dist)
    println(sum(n))
    println(sum(x_dist.vals))
    println(N)
    CSV.write("rev2_orig_bilinear/" * state * "_x_dist.csv", x_dist, writeheader=true, header=["FIPS", "x"])
    # for i in 1:T
    #     println(i)
    #     for j in 1:len
    #         println(model_data[j, "COUNTY"]  * ": " * string(value(d_bar[j, i])*100000))
    #     end
    # end
    bud_dist = Dict()
    for j in 1:len
        bud_dist[string(model_data[j, "FIPS"])] = round.(value.(d_bar[j, :]).*100000, digits = 2)
    end
    # print(bud_dist)
    bud_df = DataFrame(bud_dist)
    df_head = DataFrame([1:T], [:FIPS])
    bud_df = hcat(df_head, bud_df)
    bud_df = permutedims(bud_df, 1, strict=false)
    println(bud_df)
    println(sum(bud_df[!, "1"]))
    println(sum(bud_df[!, "8"]))
    println(d_lim[1])
    CSV.write("rev2_orig_bilinear/" * state * "_bud_dist.csv", bud_df)
    vals = []
    push!(vals, objective_value(model_nc))
    push!(vals, MOI.get(model_nc, MOI.RelativeGap()))
    push!(vals, value(h))
    push!(vals, solve_time(model_nc))
    global msrs = hcat(msrs, vals)
    # println(A_pred./scale)
    # println(R_pred./scale)
    # println(D_pred./scale)
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

CSV.write("rev2_orig_bilinear/final_sols.csv", Tables.table(cat), writeheader = true, header = states)
CSV.write("rev2_orig_bilinear/model_msrs_new.csv", Tables.table(msrs), writeheader = true, header = states)
# CSV.write("rev2_orig_relaxation/zeta_over_time.csv", Tables.table(zeta_time), writeheader = true, header = states)