using QuantumCollocation
using NamedTrajectories
using TrajectoryIndexingUtils

using CairoMakie
using DelimitedFiles
using Distributions
using LinearAlgebra

# Operators 
const n_levels = 2
at = create(n_levels)
a = annihilate(n_levels)

H_operators = Dict(
        "X" => a + at,
        "Y" => -im * (a - at),
        "Z" => I - 2 * at * a,
)

# Time
T = 50
# Δt = 0.2
Δt = 2/9


H_drift = zeros(n_levels^4, n_levels^4)
H_controls = [
    kron_from_dict("XIII", H_operators),
    kron_from_dict("IXII", H_operators),
    kron_from_dict("IIXI", H_operators),
    kron_from_dict("IIIX", H_operators),
]

X_gate = GATES[:X]
SX_gate = sqrt(GATES[:X])
U_goal = reduce(kron, [X_gate, X_gate, SX_gate, SX_gate])


H_crosstalk = (
    kron_from_dict("ZZII", H_operators)
    + kron_from_dict("IZZI", H_operators)
    + kron_from_dict("IIZZ", H_operators)
    + kron_from_dict("ZIIZ", H_operators)
    + kron_from_dict("ZIII", H_operators)
    + kron_from_dict("IZII", H_operators)
    + kron_from_dict("IIZI", H_operators)
    + kron_from_dict("IIIZ", H_operators)
)

@views function infidelity_robustness(Hₑ::AbstractMatrix, p::QuantumControlProblem)
    Z⃗ = vec(p.trajectory.data)
    Z = p.trajectory
    return InfidelityRobustnessObjective(Hₑ, Z).L(Z⃗, Z)
end

## Original control problem
# ================================

prob = UnitarySmoothPulseProblem(
    H_drift,
    H_controls,
    U_goal,
    T,
    Δt;
    a_guess=(π/2/T/Δt) * ones((length(H_controls), T)),
    timesteps_all_equal=true,
    free_time=false,
    hessian_approximation=true,
    pade_order=20,
    R=10.,
    R_dda=5.,
)

solve!(prob; max_iter=75)

println("Fidelity: ", unitary_fidelity(prob))
println("Crosstalk robustness: ", infidelity_robustness(H_crosstalk, prob))

save("saved-pulses/single_qubit_gateset_default.jld2", prob.trajectory)
writedlm("saved-pulses/a_single_qubit_gateset_default.csv", prob.trajectory[:a], ",")

## 

function random_a_guess(traj::NamedTrajectory)
    # Positive (symmetric) upper bounds
    a_bounds = traj.bounds[:a][2]

    a_dists = [Uniform(
        -(a_bounds[i] == Inf ? 1.0 : a_bounds[i]),
        (a_bounds[i] == Inf ? 1.0 : a_bounds[i])
    ) for i = 1:traj.dims[:a]]

    a = hcat([
        zeros(traj.dims[:a]),
        vcat([rand(a_dists[i], 1, T - 2) for i = 1:traj.dims[:a]]...),
        zeros(traj.dims[:a])
    ]...)
    return a
end

## Robustness to crosstalk: Initializer
# ================================

trajectory = copy(prob.trajectory)
# Need random initial conditions to avoid local minima

# Option 1:
# ----------------
# Don't stagger! Try to use the same order of magnitude as the initial gates.
# update!(trajectory, :a, [π/2/10 , π/2/10, π/4/10, π/4/10] .* random_a_guess(trajectory))
# update!(trajectory, :Ũ⃗, 2 * rand(trajectory.dims[:Ũ⃗], T) .- 1)

# Option 2:
# ----------------
# Stagger as in the plot above
update!(trajectory, :a, [1 + π, 1, 1 + π, 1] .* prob.trajectory[:a])
update!(trajectory, :Ũ⃗, 2 * rand(trajectory.dims[:Ũ⃗], T) .- 1)

parameters = deepcopy(prob.params)

objective = DefaultObjective()
objective += Objective(parameters[:objective_terms][1])
objective += QuadraticRegularizer(:dda, trajectory, 1e-3)
objective += QuadraticRegularizer(:a, trajectory, 1e-6)

update_bound!(trajectory, :a, 1.0)
constraints = trajectory_constraints(trajectory)

ipopt_options = Options()
ipopt_options.hessian_approximation = "limited-memory"

rob_prob_1 = UnitaryRobustnessProblem(
    H_crosstalk,
    trajectory,
    prob.system,
    objective,
    prob.integrators,
    constraints;
    final_fidelity=0.9, 
    verbose=false,
    build_trajectory_constraints=false,
    hessian_approximation=true,
    ipopt_options=ipopt_options
)

solve!(rob_prob_1; max_iter=100)

println("Fidelity: ", unitary_fidelity(rob_prob_1))
println("Robustness: ", infidelity_robustness(H_crosstalk, rob_prob_1))

## Robustness to crosstalk: Refiner 1
# ================================

trajectory = copy(rob_prob_1.trajectory)
parameters = deepcopy(prob.params)

objective = DefaultObjective()
objective += QuadraticRegularizer(:dda, trajectory, 1e-5)
objective += QuadraticRegularizer(:a, trajectory, 1e-5)

update_bound!(trajectory, :a, Inf)
update_bound!(trajectory, :dda, Inf)
constraints = trajectory_constraints(trajectory)

ipopt_options = Options()
ipopt_options.hessian_approximation = "limited-memory"

rob_prob_2 = UnitaryRobustnessProblem(
    H_crosstalk,
    trajectory,
    prob.system,
    objective,
    prob.integrators,
    constraints;
    final_fidelity=0.9999,
    verbose=false,
    build_trajectory_constraints=false,
    hessian_approximation=true,
    ipopt_options=ipopt_options
)

solve!(rob_prob_2; max_iter=200)

println("Fidelity: ", unitary_fidelity(rob_prob_2))
println("Robustness: ", infidelity_robustness(H_crosstalk, rob_prob_2))

save("saved-pulses/single_qubit_gateset_R1e-5.jld2", rob_prob_2.trajectory)
writedlm("saved-pulses/a_single_qubit_gateset_R1e-5.csv", prob.trajectory[:a], ",")

## Robustness to crosstalk: Refiner 2
# ================================

trajectory = copy(rob_prob_1.trajectory)
# Need random initial conditions to avoid local minima
# update!(trajectory, :a, random_a_guess(trajectory))
# update!(trajectory, :Ũ⃗, 2 * rand(trajectory.dims[:Ũ⃗], T) .- 1)
parameters = deepcopy(prob.params)

objective = DefaultObjective()
objective += QuadraticRegularizer(:dda, trajectory, 1e-3)
objective += QuadraticRegularizer(:a, trajectory, 1e-3)

update_bound!(trajectory, :a, Inf)
update_bound!(trajectory, :dda, Inf)
constraints = trajectory_constraints(trajectory)

ipopt_options = Options()
ipopt_options.hessian_approximation = "limited-memory"

rob_prob_3 = UnitaryRobustnessProblem(
    H_crosstalk,
    trajectory,
    prob.system,
    objective,
    prob.integrators,
    constraints;
    final_fidelity=0.9999,
    verbose=false,
    build_trajectory_constraints=false,
    hessian_approximation=true,
    ipopt_options=ipopt_options
)

solve!(rob_prob_3; max_iter=200)

println("Fidelity: ", unitary_fidelity(rob_prob_3))
println("Robustness: ", infidelity_robustness(H_crosstalk, rob_prob_3))

save("saved-pulses/single_qubit_gateset_R1e-3.jld2", rob_prob_3.trajectory)
writedlm("saved-pulses/a_single_qubit_gateset_R1e-3.csv", rob_prob_3.trajectory[:a], ",")


## Robustness to crosstalk: Refiner 3
trajectory = copy(rob_prob_1.trajectory)
parameters = deepcopy(prob.params)

objective = DefaultObjective()
objective += QuadraticRegularizer(:dda, trajectory, 1e-6)
objective += QuadraticRegularizer(:a, trajectory, 1e-6)

update_bound!(trajectory, :a, Inf)
update_bound!(trajectory, :dda, Inf)
constraints = trajectory_constraints(trajectory)

ipopt_options = Options()
ipopt_options.hessian_approximation = "limited-memory"

rob_prob_4 = UnitaryRobustnessProblem(
    H_crosstalk,
    trajectory,
    prob.system,
    objective,
    prob.integrators,
    constraints;
    final_fidelity=0.9999,
    verbose=false,
    build_trajectory_constraints=false,
    hessian_approximation=true,
    ipopt_options=ipopt_options
)

solve!(rob_prob_4; max_iter=200)

println("Fidelity: ", unitary_fidelity(rob_prob_4))
println("Robustness: ", infidelity_robustness(H_crosstalk, rob_prob_4))

save("saved-pulses/single_qubit_gateset_R1e-6.jld2", rob_prob_4.trajectory)
writedlm("saved-pulses/a_single_qubit_gateset_R1e-6.csv", rob_prob_4.trajectory[:a], ",")

## Robustness to crosstalk: Refiner 4
# ================================
trajectory = copy(rob_prob_1.trajectory)
parameters = deepcopy(prob.params)

objective = DefaultObjective()
objective += QuadraticRegularizer(:dda, trajectory, 1e-2)
objective += QuadraticRegularizer(:a, trajectory, 1e-2)

update_bound!(trajectory, :a, Inf)
update_bound!(trajectory, :dda, Inf)
constraints = trajectory_constraints(trajectory)

ipopt_options = Options()
ipopt_options.hessian_approximation = "limited-memory"

rob_prob_5 = UnitaryRobustnessProblem(
    H_crosstalk,
    trajectory,
    prob.system,
    objective,
    prob.integrators,
    constraints;
    final_fidelity=0.9999,
    verbose=false,
    build_trajectory_constraints=false,
    hessian_approximation=true,
    ipopt_options=ipopt_options
)

solve!(rob_prob_5; max_iter=200)

println("Fidelity: ", unitary_fidelity(rob_prob_5))
println("Robustness: ", infidelity_robustness(H_crosstalk, rob_prob_5))

save("saved-pulses/single_qubit_gateset_R1e-2.jld2", rob_prob_5.trajectory)
writedlm("saved-pulses/a_single_qubit_gateset_R1e-2.csv", rob_prob_5.trajectory[:a], ",")

## Robustness to crosstalk: Refiner 5
# ================================

trajectory = copy(rob_prob_1.trajectory)
parameters = deepcopy(prob.params)

objective = DefaultObjective()
objective += QuadraticRegularizer(:dda, trajectory, 1e-1)
objective += QuadraticRegularizer(:a, trajectory, 1e-1)

update_bound!(trajectory, :a, Inf)
update_bound!(trajectory, :dda, Inf)
constraints = trajectory_constraints(trajectory)

ipopt_options = Options()
ipopt_options.hessian_approximation = "limited-memory"

rob_prob_6 = UnitaryRobustnessProblem(
    H_crosstalk,
    trajectory,
    prob.system,
    objective,
    prob.integrators,
    constraints;
    final_fidelity=0.9999,
    verbose=false,
    build_trajectory_constraints=false,
    hessian_approximation=true,
    ipopt_options=ipopt_options
)

solve!(rob_prob_6; max_iter=200)

println("Fidelity: ", unitary_fidelity(rob_prob_6))
println("Robustness: ", infidelity_robustness(H_crosstalk, rob_prob_6))

save("saved-pulses/single_qubit_gateset_R1e-1.jld2", rob_prob_6.trajectory)
writedlm("saved-pulses/a_single_qubit_gateset_R1e-1.csv", rob_prob_6.trajectory[:a], ",")
