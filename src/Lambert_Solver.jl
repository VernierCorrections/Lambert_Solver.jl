using LinearAlgebra


"""
Stores various parameters used to control the runtime & truncation error of the various numeric solvers called by the Lambert solving algorithm, as well as the choice of cost function when picking the best transfer orbit.

...
# Arguments
- `ϵ_Halley::Float64`: Acceptable error threshold for the Halley's method solver.
- `ϵ_House1::Float64`: Less strict acceptable error threshold for the Householder's method solver.
- `ϵ_House2::Float64`: Stricter acceptable error threshold for the Householder's method solver.
- `max_Halley::Int`: Maximum number of Halley's method iterations to be called before forced truncation.
- `max_House::Int`: Maximum number of Householder's method iterations to be called before forced truncation.
- `M_limit::Int`: Artificially caps the maximum number of revolutions around the primary considered for any given transfer orbit.
- `cost_metric::Int`: Set as 0 if the desired cost function is to use the Δ_v magnitudes, and 1 if the desired cost function is to use C3s.
- `cost_weight::Float64`: Used to blend the relative weights of the departure and arrival Δ_v magnitudes or C3s when evaluating the cost function. Set this to 1.0 to consider only the departure cost, 0.0 to only consider the arrival cost, 0.5 to consider the departure and arrival costs equally, or to some intermediate value of your choice.

...
"""
@kwdef mutable struct LambertParameters
    ϵ_Halley::Float64    = 10.0^-12
    ϵ_House1::Float64    = 10.0^-10
    ϵ_House2::Float64    = 10.0^-12
    max_Halley::Int      = 100
    max_House::Int       = 100
    M_limit::Int         = 10
    cost_metric::Int     = 1
    cost_weight::Float64 = 1.0
end


"""
    Lambert_solve(r_1, r_2, v_1, v_2, Δ_t, μ, Lambert_parameters)

Returns the relative departure and arrival Δ_v vectors for the transfer orbit which solves the given orbital boundary value problem whilst simultaneously minimizing the given user-defined cost metric.

The cost of each transfer can be determined either as a weighted sum either of the departure and arrival Δ_v magnitudes, or of the departure and arrival C3s.

...
# Arguments
- `r_1::Vector{Float64}`: Initial position vector.
- `r_2::Vector{Float64}`: Target position vector.
- `v_1::Vector{Float64}`: Initial velocity vector. 
- `v_2::Vector{Float64}`: Target velocity vector.
- `Δ_t::Float64`: Time interval between departure and arrival.
- `μ::Float64`: Standard gravitational parameter of the primary.
- `Lambert_parameters::LambertParameters`: Various parameters used for the numerical control of the solution (see the LambertParameters struct)

...
"""
function Lambert_solve(r_1, r_2, v_1, v_2, Δ_t, μ, Lambert_parameters)
    r_1::Vector{Float64}
    r_2::Vector{Float64}
    v_1::Vector{Float64}
    v_2::Vector{Float64}
    Δ_t::Float64
    μ::Float64
    Lambert_parameters::LambertParameters
    # c is the chord
    c = r_2 - r_1
    c_norm = norm(c)
    r_1norm = norm(r_1)
    r_2norm = norm(r_2)
    # s is the semiperimeter (not to be confused with the semiparameter, p, which is not used in our Lambert's problem solver)
    s = 0.5 * (r_1norm + r_2norm + c_norm)
    # T is the non-dimensional time-of-flight
    T = √((2.0 * μ) / s^3) * Δ_t;
    # Radial unit vectors:
    r̂_1 = r_1 / r_1norm
    r̂_2 = r_2 / r_2norm
    # Orbital momentum and central angle:
    h = cross(r̂_1, r̂_2)
    h /= norm(h)
    λ = √(1.0 - (c_norm / s))
    # Sometimes we have to flip the orientation around for calculations
    if ((r_1[1] * r_2[2]) - (r_1[2] * r_2[1])) < 0.0
        λ = -λ
        θ̂_1 = cross(r̂_1, h)
        θ̂_2 = cross(r̂_2, h)
    else
        θ̂_1 = cross(h, r̂_1)
        θ̂_2 = cross(h, r̂_2)
    end
    # Now we have a basis which we can use to reduce our vector-valued orbital boundary value problem into a univariate root-finding problem
    # Only the components of our velocity vectors along our 4 basis directions must be solved for
    # These 4 components can be written in terms of x & y since we know the problem geometry, and y can be expressed algebraically in terms of x
    # Therefore, all we need to do now is call our numerical solvers and find every allowed value of x:
    x_list, y_list = find_xy(λ, T, Lambert_parameters)
    γ = √(μ * s / 2.0)
    ρ = (r_1norm - r_2norm) / c_norm
    σ = √(1.0 - ρ^2)
    # Now we can find the velocity vector components:
    v_r1 = γ * (((λ * y_list) - x_list) - (ρ * ((λ * y_list) + x_list))) / r_1norm
    v_r2 = -γ * (((λ * y_list) - x_list) + (ρ * ((λ * y_list) + x_list))) / r_2norm
    v_θ1 = γ * σ * (y_list + (λ * x_list)) / r_1norm
    v_θ2 = γ * σ * (y_list + (λ * x_list)) / r_2norm
    # Now we have the velocity vectors: 
    v_departure = r̂_1 * transpose(v_r1) + θ̂_1 * transpose(v_θ1)
    v_arrival = r̂_2 * transpose(v_r2) + θ̂_2 * transpose(v_θ2)
    Δ_v_departure = v_departure .- v_1 
    Δ_v_arrival = v_arrival .- v_2
    # Cost function evaluation (to find the best x):
    if Lambert_parameters.cost_metric == 0
        cost_list = Lambert_parameters.cost_weight * sqrt.(sum(abs2.(Δ_v_departure), dims = 1)) + (1.0 - Lambert_parameters.cost_weight) * sqrt.(sum(abs2.(Δ_v_arrival), dims = 1))
    elseif Lambert_parameters.cost_metric == 1
        cost_list = Lambert_parameters.cost_weight * sum(abs2.(Δ_v_departure), dims = 1) + (1.0 - Lambert_parameters.cost_weight) * sum(abs2.(Δ_v_arrival), dims = 1)
    end
    best_trajectory = argmin(cost_list)[2]
    return Δ_v_departure[:, best_trajectory], Δ_v_arrival[:, best_trajectory]
end


"""
    find_xy(λ, T, Lambert_parameters)

Finds a parameterization for every transfer orbit that solves the given orbital boundary value problem. As many different transfer orbits may provide a valid solution, the solutions that are found are returned as lists of values.

Each individual solution is parameterized in terms of x & y, variables that stand-in for angles that can uniquely reconstruct a valid transfer orbit (see Lancaster and Blanchard). The choice of the intermediary variable x (and the non-dimensional time-of-flight T, which can be written as a fairly simple function of x) results in a more versatile, numerically stable, and computationally efficient method.

...
# Arguments
- `λ::Float64`: Parameter related to the shape of the transfer orbit, calculated from the chord and semiperimeter.
- `T::Float64`: The non-dimensional time-of-flight.
- `Lambert_parameters::LambertParameters`: Various parameters used for the numerical control of the solution (see the LambertParameters struct)

...
"""
function find_xy(λ, T, Lambert_parameters)
    λ::Float64
    T::Float64
    Lambert_parameters::LambertParameters
    # First finds the maximum number of full orbits of the primary that can be completed (M_max)
    # Also calculates the non-dimensional time-of-flight for x = 0 & M = 0 by calling the Halley solver, if needed
    M_max = floor(T / π)
    T_00 = acos(λ) + (λ * √(1.0 - λ^2))
    if T < (T_00 + M_max * π) && M_max > 0
        T_min = Halley_solve(0.0, M_max, λ, T_00, Lambert_parameters.ϵ_Halley, Lambert_parameters.max_Halley)
        if T_min > T
            M_max -= 1.0
        end
    end
    T_1 = (2.0 / 3.0) * (1.0 - λ^3)
    # Logic block for direct transfer orbit handling (trajectories that don't complete a full orbit around the primary)
    # There is only one such valid trajectory, which corresponds to a single M = 0 case
    # Trivial M = 0 case handling:
    if T == T_1
        x = 1.0
        y = 1.0
    elseif T == T_00
        x = 0.0
        y = √(1.0 - λ^2)
    # Calls Householder solver if needed for non-trivial M = 0 cases:
    else
        if T ≥ T_00
            x_0 = ((T_00 / T)^(2.0 / 3.0)) - 1.0
        elseif T < T_1
            x_0 = ((5.0 / 2.0) * (T_1 * (T_1 - T)) / (T * (1.0 - λ^5))) + 1.0
        else
            x_0 = ((T_00 / T)^(log2(T_1 / T_00))) - 1.0
        end
        x = Householder_solve(x_0, 0, λ, T, Lambert_parameters.ϵ_House1, Lambert_parameters.ϵ_House2, Lambert_parameters.max_House)
        y = √(1.0 - (λ^2 * (1.0 - x^2)))
    end
    # Block for M-times around transfer orbit handling (trajectories that make at least one full orbit around the primary)
    # For each M-times around case, there is both a "left-handed" and "right-handed" trajectory (depending on which way the transfer orbit goes around the primary)
    # In this case, no logic block is needed as all cases will directly call the Householder solver
    M_max = min(M_max, Lambert_parameters.M_limit)
    if M_max != 0
        M_list = collect(range(1, M_max), inner = [2])
        x_0_list = [
        (((((M_list * π) .+ π) / (8.0 * T)).^(2.0 / 3.0) .- 1.0) ./ ((((M_list * π) .+ π) / (8.0 * T)).^(2.0 / 3.0) .+ 1.0));
        ((((8.0 * T) ./ (M_list * π)).^(2.0 / 3.0) .- 1.0) ./ (((8.0 * T) ./ (M_list * π)).^(2.0 / 3.0) .+ 1.0))
        ]
        x_list = Householder_solve.(x_0_list, (M, ), (λ, ), (T, ), (Lambert_parameters.ϵ_House1, ), (Lambert_parameters.ϵ_House2, ), (Lambert_parameters.max_House, ))
        y_list = [y; sqrt.(1.0 .- (λ^2 * (1.0 .- x_list.^2)))]
        x_list = [x; x_list]
    else
        x_list = [x]
        y_list = [y]
    end
    return x_list, y_list
end


"""
    Halley_solve(x_0, M, λ, T_00, ϵ_Halley, max_Halley)

Uses Halley's method to solve for the minima of T(x). Returns the value of T at this minima.

Halley's method requires calculation of the second derivatives of the objection function whose roots are desired, but because the goal is to find a minima of T instead of a root of T, the objective function is the first derivative of T; therefore, calculation of the third derivative of T is required.

...
# Arguments
- `x_0::Float64`: The value of x used to initialize the iterative solver.
- `M::Int`: The number of revolutions around the primary the transfer orbit to be solved for must take.
- `λ::Float64`: Parameter related to the shape of the transfer orbit, calculated from the chord and semiperimeter.
- `T_00::Float64`: The value of the non-dimensional time-of-flight for x = 0 & M = 0. Used to initialize T.
- `ϵ_Halley::Float64`: Acceptable error threshold for the Halley's method solver.
- `max_Halley::Int`: Maximum number of Halley's method iterations to be called before forced truncation.

...
"""
function Halley_solve(x_0, M, λ, T_00, ϵ_Halley, max_Halley)
    x_0::Float64
    M::Int
    λ::Float64
    T_00::Float64
    ϵ_Halley::Float64
    max_Halley::Int
    x = x_0
    for i = 1:max_Halley
        y = √(1.0 - (λ^2 * (1.0 - x^2)))
        ψ = acos((x * y) + (λ * (1.0 - x^2)))
        if i == 1
            T = T_00 + (M * π)
        else
            T = (1.0 / (1.0 - x^2)) * (((ψ + (π * M)) / √(abs(1.0 - x^2))) - (x + (λ * y)))
        end
        T′ = ((3.0 * T * x) - 2.0 + (2.0 * λ^3 * x / y)) / (1.0 - x^2)
        T′′ = ((3.0 * T) + (5.0 * x * T′) + (2.0 * (1.0 - λ^2) * λ^3 / y^3)) / (1.0 - x^2)
        T′′′ = ((7.0 * x * T′′) + (8.0 * T′) - (6.0 * (1.0 - λ^2) * λ^5 * x / y^5)) / (1.0 - x^2)
        x_new = x - ((2.0 * T′ * T′′) / ((2.0 * T′′^2) - (T′ * T′′′)))
        if (x_new - x) ≤ ϵ_Halley
            x = x_new
            break
        end
        x = x_new
    end
    y = √(1.0 - (λ^2 * (1.0 - x^2)))
    ψ = acos((x * y) + (λ * (1.0 - x^2)))
    T_min = (1.0 / (1.0 - x^2)) * (((ψ + (π * M)) / √(abs(1.0 - x^2))) - (x + (λ * y)))
    return T_min
end


"""
    Householder_solve(x_0, M, λ, T_actual, ϵ_House1, ϵ_House2, max_House)

Uses Householder's method to solve for roots of T(x) - T_actual. Returns the value of x at this root.

...
# Arguments
- `x_0::Float64`: The value of x used to initialize the iterative solver.
- `M::Int`: The number of revolutions around the primary the transfer orbit to be solved for must take.
- `λ::Float64`: Parameter related to the shape of the transfer orbit, calculated from the chord and semiperimeter.
- `T_actual::Float64`: The actual value of the non-dimensional time-of-flight. Used to compare against T calculated as a function of x at the current iteration as an error metric.
- `ϵ_House1::Float64`: Less strict acceptable error threshold for the Householder's method solver.
- `ϵ_House2::Float64`: Stricter acceptable error threshold for the Householder's method solver.
- `max_House::Int`: Maximum number of Householder's method iterations to be called before forced truncation.

...
"""
function Householder_solve(x_0, M, λ, T_actual, ϵ_House1, ϵ_House2, max_House)
    x_0::Float64
    M::Int
    λ::Float64
    T_actual::Float64
    ϵ_House1::Float64
    ϵ_House2::Float64
    max_House::Int
    if M > 0
        ϵ = ϵ_House2
    else
        ϵ = ϵ_House1
    end
    x = x_0
    for i = 1:max_House
        y = √(1.0 - (λ^2 * (1.0 - x^2)))
        if x < 1.0
            ψ = acos((x * y) + (λ * (1.0 - x^2)))
        else
            ψ = acosh((x * y) - (λ * (x^2 - 1.0)))
        end
        T = (1.0 / (1.0 - x^2)) * (((ψ + (π * M)) / √(abs(1.0 - x^2))) - (x + (λ * y)))
        T′ = ((3.0 * T * x) - 2.0 + (2.0 * λ^3 * x / y)) / (1.0 - x^2)
        T′′ = ((3.0 * T) + (5.0 * x * T′) + (2.0 * (1.0 - λ^2) * λ^3 / y^3)) / (1.0 - x^2)
        T′′′ = ((7.0 * x * T′′) + (8.0 * T′) - (6.0 * (1.0 - λ^2) * λ^5 * x / y^5)) / (1.0 - x^2)
        objective = T - T_actual
        x_new = x - objective * ((T′^2 - (objective * T′′ / 2.0)) / ((T′ * (T′^2 - (objective * T′′))) + (T′′′ * objective^2 / 6.0)));
        if (x_new - x) ≤ ϵ
            x = x_new
            break
        end
        x = x_new
    end
    return x
end

