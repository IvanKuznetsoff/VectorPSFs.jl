# aberration_functions.jl
# Aberration phase functions for normal or tilted incidence.

"""
    Φ(sρ, n)

Computes the aberration phase for **normal incidence**.

- `sρ`: the product of normalized aperture `s` and radial coordinate `ρ` in the pupil plane.
- `n`: refractive index (e.g., 1.0 for air or the immersion index).
Returns a dimensionless phase shift due to the optical path difference.
"""
function Φ(sρ, n::Real)
    @fastmath begin
        R_2 = sρ^2
        Φv = 1/n * (1 - sqrt(1 - R_2)) - n * (1 - sqrt(1 - R_2 / n^2))
    end
    return Φv
end

"""
    Φ(sρ, ϕ, n, α)

Computes the aberration phase for **tilted incidence**.

- `sρ`: again, `s * ρ`.
- `ϕ`: azimuthal angle in the pupil plane.
- `n`: refractive index.
- `α`: tilt angle (radians).

Returns a phase shift accounting for tilt plus the normal-incidence phase.
"""
function Φ(sρ, ϕ, n::Real, α::Real)
    @fastmath begin
        cosα = cos(α)
        sinα = sin(α)
        cosϕ = cos(ϕ)
        Φv = -cosα / sqrt(n^2 - sinα^2) * (
                sinα * cosϕ * sρ + cosα * sqrt(1 - sρ^2)
             ) +
              (1 - n^2) / sqrt(n^2 - sinα^2) +
              sqrt(
                  n^2 - sinα^2 -
                  cosα^2 * sρ^2 +
                  sinα^2 * sρ^2 * cosϕ^2 +
                  2sinα * cosα * sρ * cosϕ * sqrt(1 - sρ^2)
              )
    end
    return Φv
end

# -------------------------------------------------------------------
# Overloads for single plate (apply thickness)
# -------------------------------------------------------------------

"""
    Φ(sρ, n_a, t)

OPD (µm) for a **single plane-parallel plate** at normal incidence.

- `n_a` : (relative) index of the plate
- `t`   : thickness in µm

Returns t * Φ(sρ, n_a).
"""
@inline function Φ(sρ, n_a::Real, t::Real)
    return t * Φ(sρ, n_a)
end

"""
    Φ(sρ, ϕ, n_a, t, α)

OPD (µm) for a **single plate** at tilted incidence.
"""
@inline function Φ(sρ, ϕ, n_a::Real, t::Real, α::Real)
    return t * Φ(sρ, ϕ, n_a, α)
end

# -------------------------------------------------------------------
# Overloads for a stack of plates (sum of OPDs)
# -------------------------------------------------------------------

"""
    Φ(sρ, ns, ts)

OPD (µm) for a **stack of plane-parallel plates** at normal incidence.

- `ns` : vector of (relative) indices for each layer
- `ts` : vector of thicknesses in µm (same length as `ns`)
"""
@inline function Φ(sρ, ns::AbstractVector, ts::AbstractVector)
    @assert length(ns) == length(ts)
    T = promote_type(typeof(float(sρ)), eltype(ns), eltype(ts))
    Φtot = zero(T)
    @inbounds @fastmath for i in eachindex(ns, ts)
        Φtot += ts[i] * Φ(sρ, ns[i])
    end
    return Φtot
end

"""
    Φ(sρ, ϕ, ns, ts, α)

OPD (µm) for a **stack of plates** at tilted incidence.
"""
@inline function Φ(sρ, ϕ, ns::AbstractVector, ts::AbstractVector, α::Real)
    @assert length(ns) == length(ts)
    T = promote_type(typeof(float(sρ)), eltype(ns), eltype(ts), typeof(float(α)))
    Φtot = zero(T)
    @inbounds @fastmath for i in eachindex(ns, ts)
        Φtot += ts[i] * Φ(sρ, ϕ, ns[i], α)
    end
    return Φtot
end

# Optional: keep Φ_stack as thin wrappers, if you like the name
@inline Φ_stack(sρ, ns::AbstractVector, ts::AbstractVector) =
    Φ(sρ, ns, ts)

@inline Φ_stack(sρ, ϕ, ns::AbstractVector, ts::AbstractVector, α::Real) =
    Φ(sρ, ϕ, ns, ts, α)