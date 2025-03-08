"""
Stores the six equinoctial MRP elements (Modified Rodriguez Parameters equinoctial basis elements): ł, e_1, e_2, σ_1, σ_2, λ as defined by Peterson, Arya, & Junkins (2023).

...
# Arguments
- `ł::Float64`: Semi-latus rectum, in meters.
- `e_1::Float64`: Dot product of the eccentricity vector (Laplace-Runge-Lenz vector) and the first equinoctial basis vector.
- `e_2::Float64`: Dot product of the eccentricity vector (Laplace-Runge-Lenz vector) and the second equinoctial basis vector.
- `σ_1::Float64`: Dot product of the modified Rodriguez vector and the first equinoctial basis vector.
- `σ_2::Float64`: Dot product of the modified Rodriguez vector and the second equinoctial basis vector.
- `λ::Float64`: True longitude, in radians. Normally written as ł (with the semi-latus rectum written as p), but here written as λ. This been done to avoid confusion between the semi-latus rectum and either the semiperimeter, s, or the semiparameter, p.

...
"""
@kwdef mutable struct MRPEquinoctialStatic
    ł::Float64
    e_1::Float64
    e_2::Float64
    σ_1::Float64
    σ_2::Float64
    λ::Float64
end


"""
Parameterizes an orbit in terms of an initial epoch, a final epoch, and six sets of 13 best-fit coefficients for 12th-order Chebyshev polynomials of the first kind that parameterize each equinoctial element's best fit coefficients over the resulting interval.

...
# Arguments
- `coefficients::Matrix{Float64}`: The set of best-fit coefficients for the 12th-order Chebyshev polynomials of the first kind that parameterize the orbital pertubations over the interval. 
- `t_i::Float64`: The initial epoch, in seconds.
- `t_f::Float64`: The final epoch, in seconds.

...
"""
@kwdef mutable struct EquinoctialChebyshevEphemerides
    coefficients::Matrix{Float64}
    t_i::Float64
    t_f::Float64
end


"""
Stores the six classical Keplerian orbit elements: a, e, i, Ω, ω, M

...
# Arguments
- `a::Float64`: Semi-major axis of the orbit, in meters. Use positive values for elliptical orbits, negative values for hyperbolic orbits, and use the magnitude of the radius vector at periapsis for parabolic orbits.
- `e::Float64`: Orbital eccentricity (dimensionless). Values ∈ [0, 1) describe elliptical orbits, values ϵ (1, ∞) describe hyperbolic orbits, values of 1 describe parabolic orbits.
- `i::Float64`: Orbital inclination, in radians.
- `Ω::Float64`: Longitude of the ascending node, in radians.
- `ω::Float64`: Argument of periapsis, in radians.
- `M::Float64`: Mean anomaly, in radians.

...
"""
@kwdef mutable struct ClassicalKeplerStatic
    a::Float64
    e::Float64
    i::Float64
    Ω::Float64
    ω::Float64
    M::Float64
end


"""
Parameterizes an orbit in terms of an initial epoch and the orbital elements of the orbit as determined at that epoch.

...
# Arguments
- `elements::ClassicalKeplerStatic`: The classical Keplerian orbital elements as determined at the epoch.
- `t_i::Float64`: The initial epoch, in seconds.

...
"""
@kwdef mutable struct ClassicalKeplerOrbit
    elements::ClassicalKeplerStatic
    t_i::Float64
end


"""
    ChebyshevT(x::Float64)

Generates a vector where each entry is the (n - 1)th order term of the 12th order Chebyshev polynomial of the first kind.

...
# Arguments
- `x::Float64`

...
"""
function ChebyshevT(x::Float64)
    return [
        1;
        x;
        2    * x^2  - 1;
        4    * x^3  - 3    * x;
        8    * x^4  - 8    * x^2  + 1;
        16   * x^5  - 20   * x^3  + 5    * x;
        32   * x^6  - 48   * x^4  + 18   * x^2 - 1;
        64   * x^7  - 112  * x^5  + 56   * x^3 - 7    * x;
        128  * x^8  - 256  * x^6  + 160  * x^4 - 32   * x^2 + 1; 
        256  * x^9  - 576  * x^7  + 432  * x^5 - 120  * x^3 + 9   * x;
        512  * x^10 - 1280 * x^8  + 1120 * x^6 - 400  * x^4 + 50  * x^2 - 1;
        1024 * x^11 - 2816 * x^9  + 2816 * x^7 - 1232 * x^5 + 220 * x^3 - 11 * x;
        2048 * x^12 - 6144 * x^10 + 6912 * x^8 - 3584 * x^6 + 840 * x^4 - 72 * x^2 + 1 
    ]
end


"""
    OTC(orbit, t, μ, ignore_domain::Bool = False))

Converts an orbit or a set of ephemerides, combined with an epoch, into a Cartesian-coordinate state vector.

...
# Arguments
- `orbit`: The orbit or set of ephemerides to be converted into state vector form.
- `t::Float64`: The epoch at which the state vector should be computed, in seconds.
- `μ::Float64`: Standard gravitational parameter of the primary.
- `ignore_domain::Bool`: Whether or not to allow epochs outside of the domain given by the ephemerides to be evaluated (false by default).

...
"""
function OTC(orbit::EquinoctialChebyshevEphemerides, t, μ, ignore_domain::Bool = False)
    t::Float64
    μ::Float64
    if ignore_domain == False && t < orbit.initial_epoch || t > orbit.final_epoch
        error("the epoch is not contained in the domain given by the ephemerides.")
    end
    t_elapsed = t - orbit.t_i
    Δ_t = orbit.t_f - orbit.t_i
    x = 2.0 * (t_elapsed / Δ_t) - 1.0
    temp_elements = orbit.coefficients * ChebyshevT(x)
    elements = MRPEquinoctialStatic(
        temp_elements[1],
        temp_elements[2],
        temp_elements[3],
        temp_elements[4],
        temp_elements[5],
        temp_elements[6]
    )
    S = ETC(elements, μ)
    return S
end


function OTC(orbit::ClassicalKeplerOrbit, t, μ)
    t::Float64
    μ::Float64
    if orbit.t_i != t
        t_elapsed = t - orbit.t_i
        if e > 1
            orbit.elements.M += t_elapsed * √(μ / ((-orbit.elements.a)^3))
        elseif e == 1
            orbit.elements.M += t_elapsed * √(μ / (2.0 * orbit.elements.a^3))
        else
            orbit.elements.M += t_elapsed * √(μ / (orbit.elements.a^3))
        end
    end
    S = ETC(orbit.elements, μ)
    return S
end


"""
    Kepler_solve(M, e, ϵ_Kepler::Float64 = 10.0^-12, max_Kepler::Int = 100)

Solves the inverse Kepler problem by using the Newton-Raphson method, returning an eccentric anomaly that corresponds to a given mean anomaly.

Do not call this solver for a parabolic orbit. There is no reason to anyways (the inverse Kepler problem for parabolic orbits can be solved in analytic closed-form with elementary functions).

...
# Arguments
- `M::Float64`: The mean anomaly to be converted into an eccentric anomaly.
- `e::Float64`: Orbital eccentricity.
- `ϵ_Kepler::Float64`: Acceptable error threshold for the inverse Kepler's problem solver.
- `max_Kepler::Int`: Maximum number of Newton-Raphson method iterations to be called before forced truncation.

...
"""
function Kepler_solve(M, e, ϵ_Kepler::Float64 = 10.0^-12, max_Kepler::Int = 100)
    M::Float64
    e::Float64
    if e == 1.0
        error("orbits with eccentricities of 1 (parabolic orbits) not permitted. Solve for the true anomaly analytically instead.")
    end
    E_cos = e < 1.0 ? cos : cosh
    Ê_new = M
    for i = 1:max_Kepler
        Ê_old = Ê_new
        Ê_new = Ê_old - ((Ê_old - (e * sin(Ê_old)) - M) / (1.0 - (e * E_cos(Ê_old))))
        if (Ê_new - Ê_old) ≤ ϵ_Kepler 
            break
        end
    end
    return Ê_new
end


"""
    ETC(elements, μ)

Converts a state stored as orbital elements into a state stored as a Cartesian-coordinate state vector ::Vector{Float64} = [r; v]
...
# Arguments
- `elements`: The orbital elements to be converted into a Cartesian-coordinate state vector.
- `μ::Float64`: Standard gravitational parameter of the primary.

...
"""
function ETC(elements::MRPEquinoctialStatic, μ)
    μ::Float64
    ł   = elements.ł
    e_1 = elements.e_1
    e_2 = elements.e_2
    σ_1 = elements.σ_1
    σ_2 = elements.σ_2
    λ   = elements.λ
    σ_abs2 = σ_1^2 + σ_2^2
    k = (1.0 - σ_abs2) / 2.0
    # Magnitude of the specific angular orbital momentum vector:
    h_mag = √(μ * ł)
    # Magnitude of the radius vector:
    r_mag = ł / (1.0 + e_1 * cos(λ) + e_2 * sin(λ))
    r_equinoctial = r_mag * [cos(λ); sin(λ)]
    v_equinoctial = (μ / h_mag) * [-(e_2 + sin(λ)); (e_1 + cos(λ))]
    R = (4.0 / (1.0 + σ_abs2)^2) * [
        [(k^2 + σ_1^2 - σ_2^2); (2.0 * σ_1 * σ_2); -(2.0 * k * σ_2)] [(2.0 * σ_1 * σ_2); (k^2 - σ_1^2 + σ_2^2); (2.0 * k * σ_1)]
    ]
    S = [(R * r_equinoctial); (R * v_equinoctial)]
    return S
end


function ETC(elements::ClassicalKeplerStatic, μ)
    μ::Float64
    a = elements.a
    e = elements.e
    i = elements.i
    Ω = elements.Ω
    ω = elements.ω
    M = elements.M
    # First ν, the true anomaly, must be found
    if e == 1.0
        ν = 2.0 * atan(2.0 * sinh(asinh((3.0 / 2.0) * M) / 3.0))
    else
        E = Kepler_solve(M, e)
        if e < 1.0
            ν = 2.0 * atan(√((1.0 + e) / (1.0 - e)) * tan(E / 2.0))
        else
            ν = 2.0 * atan(√((e + 1.0) / (e - 1.0)) * tanh(E / 2.0))
        end
    end
    # From this we can find the in-plane radius vector and the perpendicular vector to the radius vector
    r_plane = [cos(ν); sin(ν); 0.0]
    r_perp = cross([0.0; 0.0; 1.0], r_plane)
    # We can now use the true anomaly and eccentricity to find ϕ, the flight-path angle
    ϕ = (e * sin(ν)) / (1.0 + e * cos(ν))
    # The flight-path angle can now be used to rotate the perpendicular vector to the radius vector into the velocity vector using a transformation matrix
    v_plane = [cos(-ϕ) -sin(-ϕ) 0.0; sin(-ϕ) cos(-ϕ) 0.0; 0.0 0.0 1.0] * r_perp
    # Then simply construct a transformation matrix that rotates the in-plane vectors into the final radius and velocity vectors
    rot_mat = [cos(Ω) (-sin(Ω) * cos(i)) (sin(Ω)*sin(i)); sin(Ω) (cos(Ω)*cos(i)) -(cos(Ω)*sin(i)); 0.0 sin(i) cos(i)] * [cos(ω) -sin(ω) 0.0; sin(ω) cos(ω) 0.0; 0.0 0.0 1.0]
    # Lastly, find the magnitudes of the velocity and radius vectors
    # ł, the semi-latus rectum, is nedeed for this
    if e == 1.0
        ł = 2.0 * a
    else
        ł = a * (1.0 - e^2)
    end
    # This gives us the magnitude of the radius vector
    r_mag = ł / (1.0 + e * cos(ν))
    # Using the vis-visa equation, we can find the magnitude of the velocity vector
    if e == 1
        v_mag = √(μ * (2.0 / r_mag))
    else
        v_mag = √(μ * ((2.0 / r_mag) - (1.0 / a)))
    end
    # Now just multiply everything together
    S = [(r_mag * (rot_mat * r_plane)); (v_mag * (rot_mat * v_plane))]
    return S
end



"""
    KTM(elements)

Converts a state stored as Keplerian orbital elements ::ClassicalKeplerStatic into a state stored as MRP equinoctial orbital elements ::MRPEquinoctialStatic.  

...
# Arguments
- `elements::ClassicalKeplerStatic`: The Keplerian orbital elements to be converted into MRP equinoctial orbital elements.

...
"""
function KTM(elements)
    elements::ClassicalKeplerStatic
    a = elements.a
    e = elements.e
    i = mod2pi(elements.i)
    Ω = elements.Ω
    ω = elements.ω
    M = elements.M
    # The first element is the semi-latus rectum
    if e == 1.0
        ł = 2.0 * a
    else
        ł = a * (1.0 - e^2)
    end
    # Eccentricity vector components:
    e_1 = e * cos(Ω + ω)
    e_2 = e * sin(Ω + ω)
    # Rodriguez vector components:
    σ_1 = tan(i / 4.0) * cos(Ω)
    σ_2 = tan(i / 4.0) * sin(Ω)
    # True anomaly calculation:
    if e == 1.0
        ν = 2.0 * atan(2.0 * sinh(asinh((3.0 / 2.0) * M) / 3.0))
    else
        E = Kepler_solve(M, e)
        if e < 1.0
            ν = 2.0 * atan(√((1.0 + e) / (1.0 - e)) * tan(E / 2.0))
        else
            ν = 2.0 * atan(√((e + 1.0) / (e - 1.0)) * tanh(E / 2.0))
        end
    end
    # True longitude (normally written as ł, here written as λ) calculation from true anomaly:
    λ = Ω + ω + ν
    MRP_equinoctial_elements = MRPEquinoctialStatic(ł, e_1, e_2, σ_1, σ_2, λ)
    return MRP_equinoctial_elements
end


"""
    CTK(S, μ)

Converts a state stored as a Cartesian-coordinate state vector ::Vector{Float64} = [r; v] into a state represented by the 6 classical Keplerian orbital elements ::ClassicalKeplerStatic.  

...
# Arguments
- `S::Vector{Float64}`: The state vector to be converted into the 6 classical Keplerian orbital elements.
- `μ::Float64`: Standard gravitational parameter of the primary.

...
"""
function CTK(S, μ)
    S::Vector{Float64}
    μ::Float64
    r = S[1:3]
    v = S[4:6]
    r_mag = sqrt(sum(abs2.(r), dims = 1)[1])
    v_mag = sqrt(sum(abs2.(v), dims = 1)[1])
    # Find the specific orbital angular momentum vector, which defines the normal vector of the orbital plane
    h = cross(r, v)
    # Find the semi-latus rectum ł, which can be used to find the semi-major axis a and eccentricity e
    ł = sum(abs2.(h), dims = 1)[1] / μ
    if v_mag == √(2.0 * μ / r_mag)
        a = ł / 2.0
        e = 1.0
    else
        a = -(1.0 / ((v_mag^2 / μ) - (2.0 / r_mag)))
        e = √(1.0 - (ł / a))
    end
    # The x and y components of the specific orbital momentum vector directly give us the longitude of the ascending node
    h_x = h[1]
    h_y = h[2]
    Ω = atan(h_x, -h_y)
    # The orientation of the specific orbital momentum vector also gives us the eccentricity
    z_proj = dot([0.0; 0.0; 1.0], h)
    p = -cross([0.0; 0.0; 1.0], [cos(Ω), sin(Ω), 0.0])
    p_proj = dot(p, h)
    i = atan(p_proj, z_proj)
    # A number of different cases must be considered for singularity-free calculation of the true anomaly ν and the argument of periapsis ω
    if (i == 0.0) && (e == 0.0)
        # This is actually the true longitude, normally written as ł but here written as λ, not the true anomaly ν
        # The argument of periapsis and longitude of the ascending node are both undefined, as the orbit is both uninclined and circular, so both are taken to be 0 for convenience
        # λ = ν + Ω + ω, but since Ω and ω have been taken to be 0, we can just say λ = v
        ν = atan(r[2], r[1])
        ω = 0.0
    elseif e == 0.0
        # This is actually u, the argument of latitude, not the true anomaly ν
        # The argument of periapsis is undefined, as the orbit is circular, so it is taken to be 0 for convenience
        # u = ν + ω, but since ω has been taken to be 0, we can just say u = ν
        ν = atan((r[3] / sin(i)), (r[1] * cos(Ω) + r[2] * sin(Ω)))
        ω = 0.0
    else
        ν = atan((√(ł / μ) * dot(v, r)), (ł - r_mag))
        # u is the argument of latitude
        u = atan((r[3] / sin(i)), (r[1] * cos(Ω) + r[2] * sin(Ω)))
        ω = u - ν
    end
    # Lastly, we find the mean anomaly
    if e == 1.0
        M = (2.0 / 3.0) * sinh(3.0 * asinh(tan(ν / 2.0) / 2.0))
    elseif e < 1
        E = 2.0 * atan(√((1.0 - e) / (1.0 + e)) * tan(ν / 2.0))
        M = E - e * sin(E)
    else
        E = 2.0 * atanh(√((e - 1.0) / (e + 1.0)) * tan(ν / 2.0))
        M = e * sin(E) - E
    end
    i = i < 0.0 ? (2.0 * π) + i : i
    Ω = Ω < 0.0 ? (2.0 * π) + Ω : Ω
    ω = ω < 0.0 ? (2.0 * π) + ω : ω
    M = M < 0.0 ? (2.0 * π) + M : M
    elements = ClassicalKeplerStatic(a, e, i, Ω, ω, M)
    return elements
end


"""
    CTM(S, μ)

Converts a state stored as a Cartesian-coordinate state vector ::Vector{Float64} = [r; v] into a state represented by the 6 MRP equinoctial orbital elements ::MRPEquinoctialStatic.  

...
# Arguments
- `S::Vector{Float64}`: The state vector to be converted into the 6 MRP equinoctial orbital elements.
- `μ::Float64`: Standard gravitational parameter of the primary.

...
"""
function CTM(S, μ)
    # Mostly the same algorithm as CTK
    S::Vector{Float64}
    μ::Float64
    r = S[1:3]
    v = S[4:6]
    r_mag = sqrt(sum(abs2.(r), dims = 1)[1])
    v_mag = sqrt(sum(abs2.(v), dims = 1)[1])
    # Find the specific orbital angular momentum vector, which defines the normal vector of the orbital plane
    h = cross(r, v)
    # Find the semi-latus rectum ł, which can be used to find the semi-major axis a and eccentricity e
    ł = sum(abs2.(h), dims = 1)[1] / μ
    if v_mag == √(2.0 * μ / r_mag)
        a = ł / 2.0
        e = 1.0
    else
        a = -(1.0 / ((v_mag^2 / μ) - (2.0 / r_mag)))
        e = √(1.0 - (ł / a))
    end
    # The x and y components of the specific orbital momentum vector directly give us the longitude of the ascending node
    h_x = h[1]
    h_y = h[2]
    Ω = atan(h_x, -h_y)
    # The orientation of the specific orbital momentum vector also gives us the eccentricity
    z_proj = dot([0.0; 0.0; 1.0], h)
    p = -cross([0.0; 0.0; 1.0], [cos(Ω), sin(Ω), 0.0])
    p_proj = dot(p, h)
    i = atan(p_proj, z_proj)
    # Diverges from CTK here, since we can just directly calculate the true longitude:
    if (i == 0.0) && (e == 0.0)
        # True longitude calculation for this case is the easiest
        λ = atan(r[2], r[1])
        # The argument of periapsis is undefined, as there is no periapsis, but we can take it to be zero for convenience
        ω = 0.0
    elseif e == 0.0
        # For this case we first get the argument of latitude u, where u = ν + ω
        u = atan((r[3] / sin(i)), (r[1] * cos(Ω) + r[2] * sin(Ω)))
        # λ = ν + Ω + ω, so λ = u + Ω:
        λ = u + Ω
        # The argument of periapsis is undefined, as there is no periapsis, but we can take it to be zero for convenience
        ω = 0.0
    else
        # In the general case we still first calculate the argument of latitude u and use it to obtain the true anomaly
        u = atan((r[3] / sin(i)), (r[1] * cos(Ω) + r[2] * sin(Ω)))
        λ = u + Ω
        # However, we also need to find the true anomaly ν in order to find the argument of periapsis
        ν = atan((√(ł / μ) * dot(v, r)), (ł - r_mag))
        # Recall u = ν + ω, therefore ω = u - ν
        ω = u - ν
    end
    # Eccentricity vector components:
    e_1 = e * cos(Ω + ω)
    e_2 = e * sin(Ω + ω)
    # Rodriguez vector components:
    σ_1 = tan(i / 4.0) * cos(Ω)
    σ_2 = tan(i / 4.0) * sin(Ω)
    MRP_equinoctial_elements = MRPEquinoctialStatic(ł, e_1, e_2, σ_1, σ_2, λ)
    return MRP_equinoctial_elements
end

