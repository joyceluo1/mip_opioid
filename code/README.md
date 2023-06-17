## File descriptions and pipeline

### Descriptions:
1. indiv_state_train_updated.jl: estimate compartmental model parameters using neural ODEs
2. original_bilinear_model.jl: implementation of original bilinear MIP using JuMP and Gurobi
3. open_close_bilinear_model.jl: implementation of open/close bilinear MIP using JuMP and Gurobi
4. original_relaxation.jl: implementation of our strong relaxation using McCormick envelopes to obtain high-quality solutions to the original MIP
5. open_close_relaxation.jl: implementation of our strong relaxation using McCormick envelopes to obtain high-quality solutions to the open/close MIP
6. original_upper_bound.jl: get upper bound on optimal objective value of original MIP and propagate dynamics using feasible solution
7. open_close_upper_bound.jl: get upper bound on optimal objective value of open/close MIP and propagate dynamics using feasible solution

### Pipeline:
Our solution method for original formulation:
1. indiv_state_train_updated.jl
2. original_relaxation.jl
3. original_upper_bound.jl

Our solution method for open/close formulation:
1. indiv_state_train_updated.jl
2. open_close_relaxation.jl
3. open_close_upper_bound.jl
