#=

Adapted from the following GitHub repo code: https://github.com/khannay/FittingParamsDiffEqFlux/tree/master/julia

Original code citations: 
https://github.com/SciML/DiffEqFlux.jl#optimizing-parameters-of-an-ode-for-an-optimal-control-problem
https://julialang.org/blog/2019/01/fluxdiffeq/
https://docs.sciml.ai/DiffEqFlux/stable/examples/neural_ode/

Fits model parameters using neural ODEs for a set of US states.

=#

using DifferentialEquations, Lux, Optim, Optimization, OptimizationOptimJL, DiffEqFlux, Plots, CSV, DataFrames, DelimitedFiles, Sundials, Tables, Statistics

tstart = 0.0
tend = 20.0
sampling = 1

model_params = [sqrt(0.3), sqrt(0.9), sqrt(0.19), sqrt(0.5), sqrt(0.01159)]

# take square root of all the numbers and then square them in the model
alpha_sr = sqrt(0.15)
gamma_sr = sqrt(0.00744)
delta_sr = sqrt(0.1)
sigma_sr = sqrt(0.9)

function model(du, u, p, t)
    S, P, I, A, R, D = u
    
    phi_sr, epsilon_sr, beta_sr, zeta_sr, mu_sr = p

    du[1] = -(alpha_sr^2)*S + (epsilon_sr^2)*P + (delta_sr^2)*R
    du[2] = (alpha_sr^2)*S - ((epsilon_sr^2) + (gamma_sr^2) + (beta_sr^2))*P 
    du[3] = (beta_sr^2)*P - (phi_sr^2)*I
    du[4] = (gamma_sr^2)*P + (sigma_sr^2)*R + (phi_sr^2)*I - (zeta_sr^2)*A - (mu_sr^2)*A
    du[5] = (zeta_sr^2)*A - ((delta_sr^2) + (sigma_sr^2))*R
    du[6] = (mu_sr^2)*A

end

states = ["AL", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI",
        "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN",
        "MO", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "OH", "OK",
       "OR", "PA", "RI", "SC", "TN", "TX", "UT", "VT", "VA", "WA",
       "WI"]

global cat = Array{Float64}(undef, 5, 0)
errors = DataFrames.DataFrame(State = String[], S_avg_percent_error = Float64[], P_avg_percent_error = Float64[],
    I_avg_percent_error = Float64[], A_avg_percent_error = Float64[], R_avg_percent_error = Float64[], 
    D_avg_percent_error = Float64[])
errors_pop = DataFrames.DataFrame(State = String[], S_avg_percent_error = Float64[], P_avg_percent_error = Float64[],
    I_avg_percent_error = Float64[], A_avg_percent_error = Float64[], R_avg_percent_error = Float64[], 
    D_avg_percent_error = Float64[])

for i in 1:length(states)
    data = readdlm("tuples/yearly_tuples_prop_" * states[i] * ".csv", ',', Float64)
    Q = ifelse.(data .> 0, 1, data)
    # average scaling 
    avg_scale = sum(data, dims=1)./sum(Q, dims=1)
    for z in 1:6
        Q[:, z] = Q[:, z]./avg_scale[z]
    end
    u0 = data[1, :]
    
    function predict_adjoint(param) # Our 1-layer neural network
        prob=ODEProblem(model,u0,(tstart,tend), model_params)
        Array(concrete_solve(prob, Tsit5(), u0, param, saveat=tstart:sampling:tend, 
            abstol=1e-9,reltol=1e-9, sensealg = ForwardDiffSensitivity()))
    end
    
    function loss_adjoint(param)
        prediction = predict_adjoint(param)
        prediction_t = prediction'

        loss = sum(abs2, Q .* (prediction_t - data)) 
        loss
    end

    losses = []
    callback(θ,l) = begin
        push!(losses, l)
        # if length(losses)%50==0
        #     println(losses[end])
        # end
        false
    end

    function plotFit(param, u0, st)

        tspan=(tstart,tend)
        sol_fit=solve(ODEProblem(model,u0, tspan, param), Tsit5(), saveat=tstart:sampling:tend)

        tgrid=tstart:sampling:tend
        pl=plot(sol_fit, idxs=[2 3 4 5 6], lw=2, legend=:outertopright, label = ["P" "I" "A" "R" "D"])
        scatter!(pl,tgrid, exp10.(data[:,2]), color=:blue, label = "P")
        scatter!(pl,tgrid, exp10.(data[:,3]), color=:orange, label = "I")
        scatter!(pl,tgrid, exp10.(data[:,4]), color=:green, label = "A")
        scatter!(pl,tgrid, exp10.(data[:,5]), color=:pink, label = "R")
        scatter!(pl,tgrid, exp10.(data[:,6]), color=:brown, label = "D")
        xlabel!(pl,"Time")
        ylabel!(pl,"Population Proportion")
        # savefig(pl, "figs/fitdyn_" * st * ".pdf")
        display(pl)
        return(Array(sol_fit))
    end

    
    function train_model()
        pguess=[sqrt(0.3), sqrt(0.9), sqrt(0.19), sqrt(0.5), sqrt(0.01159)]
    #     println("Losses (every 50 iters):")
    #     println("$(loss_adjoint(pguess)[1])")
        # train the ODE
        optf = OptimizationFunction((x, p) -> loss_adjoint(x), Optimization.AutoZygote())
        optprob = Optimization.OptimizationProblem(optf, pguess)
        result_neuralode = Optimization.solve(optprob, ADAM(0.0001), callback = callback, maxiters = 25000) # initial 0.0001
        optprob2 = remake(optprob, u0 = result_neuralode.u)
        result_neuralode2 = Optimization.solve(optprob2, BFGS(initial_stepnorm = 1e-5), callback = callback) # initial 1e-5
        println("Fitted parameters:")
        println("$((result_neuralode2.minimizer).^2)")
        return(result_neuralode2)
    end
    
    println(states[i])
    res_norm = train_model()
    result = (res_norm.minimizer).^2
    global cat = hcat(cat, result) 
    model_preds = plotFit(res_norm.minimizer, u0, states[i])
    # println(model_preds)
    ls = plot(losses, xlabel = "Iterations", ylabel = "Loss", yaxis = :log, legend = false)
    # savefig(ls, "figs/loss_" * states[i] * ".pdf")
    display(ls)

    # Compare data and model predictions
    percent_errors = []
    percent_errors_pop = []
    for i in 1:6
        num_nz = 0
        sum_pe = 0
        sum_pe_pop = 0
        for j in 1:length(data[:, i])
            if data[j, i] != 0.0
                sum_pe += (abs(model_preds[i, j] - data[j, i])/data[j, i])*100
                sum_pe_pop += (abs(model_preds[i, j] - data[j, i]))*100
                num_nz += 1
            end
        end
        avg_percent_error = sum_pe/num_nz
        push!(percent_errors, avg_percent_error)
        avg_percent_error_pop = sum_pe_pop/num_nz
        push!(percent_errors_pop, avg_percent_error_pop)
    end
    push!(errors, (states[i], percent_errors[1], percent_errors[2], 
        percent_errors[3], percent_errors[4], percent_errors[5], percent_errors[6]))
    push!(errors_pop, (states[i], percent_errors_pop[1], percent_errors_pop[2], 
        percent_errors_pop[3], percent_errors_pop[4], percent_errors_pop[5], percent_errors_pop[6]))
end

CSV.write("avg_weighting_sp.csv", Tables.table(cat), writeheader = true, header = states)
# println(errors)
# println(errors_pop)
CSV.write("avg_weighting_errors_new.csv", errors)
CSV.write("avg_weighting_errors_pop.csv", errors_pop)

# Reformatting data
param = DataFrame(CSV.File("avg_weighting_sp.csv"))
cols = names(param)
param = Matrix(param)
param = permutedims(param, (2, 1))
new_param_df = DataFrame(param, ["phi", "epsilon", "beta", "zeta", "mu"])
new_param_df[!, "State"] = cols
CSV.write("avg_weighting_state_params.csv", new_param_df)