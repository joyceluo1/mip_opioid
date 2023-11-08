## File descriptions and pipeline

### Descriptions:
1. indiv_state_train_weighted.jl: estimate compartmental model parameters using neural ODE-inspired model fitting process and average-weighted loss
2. bilinear_formulation.jl: implementation of original bilinear MIP using JuMP and Gurobi
3. relaxed_formulation.jl: implementation of our strong relaxation using McCormick envelopes to obtain high-quality solutions to the original MIP
4. get_upper_bound.jl: get upper bound on optimal objective value of original MIP and propagate dynamics using feasible solution

### Pipeline:
Our solution method:
1. indiv_state_train_weighted.jl
2. relaxed_formulation.jl
3. get_upper_bound.jl
