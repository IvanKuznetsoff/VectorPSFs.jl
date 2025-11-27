# psf_core.jl
# Core computations for vector PSF integrals, handling aberration-free,
# normal incidence, or tilted incidence through plane-parallel plates.

"""
    Ξ(ρ, ϕ, ψ, u, v, s, n)

Computes the phase term for the aberration-free case.  

- `ρ, ϕ` are polar coordinates in the pupil.
- `ψ` is an azimuthal shift (e.g. relating to off-axis propagation).
- `u, v` are defocus/piston terms in the exponent.
- `s` is the normalized aperture (NA).
- `n` is the refractive index (often 1.0 for air or `obj.n` for immersion).
"""
function Ξ(ρ, ϕ, ψ, u, v, s, n)
    @fastmath Ξv = v * ρ * cos(ϕ - ψ) + u * sqrt(1 - s^2 * ρ^2 / n^2)
end

"""
    ξx(ρ, ϕ, Ξ, s)

x-polarized integrand for the aberration-free PSF integral.  
- `ρ, ϕ` are pupil-plane coordinates.
- `Ξ` is the accumulated phase from `Ξ(...)`.
- `s` is the normalized aperture.
Returns the complex field contribution in the x-direction.
"""
function ξx(ρ, ϕ, Ξ, s)
    @fastmath begin
        R_2 = s^2 * ρ^2
        res = exp(im * Ξ) * ρ * s^2 *
              ( sqrt(1 - R_2) +
                (1 - sqrt(1 - R_2)) * (1 - cos(2ϕ)) / 2
              ) / (1 - R_2)^(1/4)
    end
    return res
end

"""
    ξy(ρ, ϕ, Ξ, s)

y-polarized integrand for the aberration-free PSF integral.
Similar arguments as `ξx(...)`.
"""
function ξy(ρ, ϕ, Ξ, s)
    @fastmath begin
        R_2 = s^2 * ρ^2
        res = exp(im * Ξ) * ρ * s^2 *
              (1 - sqrt(1 - R_2)) * sin(2ϕ) / 2 / (1 - R_2)^(1/4)
    end
    return res
end

"""
    ξz(ρ, ϕ, Ξ, s)

z-polarized integrand for the aberration-free PSF integral.
Similar arguments as `ξx(...)`.
"""
function ξz(ρ, ϕ, Ξ, s)
    @fastmath begin
        R_2 = s^2 * ρ^2
        res = exp(im * Ξ) * R_2 * s * cos(ϕ) / 2 / (1 - R_2)^(1/4)
    end
    return res
end

"""
    psf_inner(ψ, u, v, s, n; rtol=1e-4, atol=1e-5)

Computes the aberration-free PSF intensity by numerically integrating
the x-, y-, and z-polarized integrands via `hcubature`.

- `ψ, u, v` are phase parameters relating to off-axis, defocus, etc.
- `s` is the normalized aperture.
- `n` is the refractive index (1.0 or immersion).
- `rtol`, `atol` are integration tolerances.
"""
function psf_inner(ψ, u, v, s, n; rtol=1e-4, atol=1e-5)
    resx = hcubature(
        dr -> ξx(dr[1], dr[2], Ξ(dr[1], dr[2], ψ, u, v, s, n), s),
        [0.0, 0.0], [1.0, 2π],
        rtol=rtol, atol=atol
    )
    resy = hcubature(
        dr -> ξy(dr[1], dr[2], Ξ(dr[1], dr[2], ψ, u, v, s, n), s),
        [0.0, 0.0], [1.0, 2π],
        rtol=rtol, atol=atol
    )
    resz = hcubature(
        dr -> ξz(dr[1], dr[2], Ξ(dr[1], dr[2], ψ, u, v, s, n), s),
        [0.0, 0.0], [1.0, 2π],
        rtol=rtol, atol=atol
    )

    ret = 0.0
    ret += abs2(resx[1]::ComplexF64)
    ret += abs2(resy[1]::ComplexF64)
    ret += abs2(resz[1]::ComplexF64)
    return ret / π^2
end

# -------------------------------------------------------------------
# Normal incidence with a plane-parallel plate or plate stack
# -------------------------------------------------------------------

"""
    Ξ(ρ, ϕ, ψ, u, v, k, s, n, n_a, t)

Phase term for normal incidence with a plane-parallel plate **or stack**.

- `n_a` : either a scalar (single plate) or vector of relative indices
- `t`   : scalar thickness or vector of thicknesses (µm)

Adds the plate-induced phase `k * Φ(sρ, n_a, t)` to the base aberration-free phase.
"""
function Ξ(ρ, ϕ, ψ, u, v, k, s, n, n_a, t)
    Φv = Φ(s * ρ, n_a, t)               # <- multiple dispatch handles scalar vs vector
    @fastmath Ξv = k * Φv + Ξ(ρ, ϕ, ψ, u, v, s, n)
end

"""
    ξx(ρ, ϕ, ψ, u, v, k, s, n, n_a, t)

x-polarized integrand with normal-incidence plate (single or stacked).
"""
function ξx(ρ, ϕ, ψ, u, v, k, s, n, n_a, t)
    Ξv = Ξ(ρ, ϕ, ψ, u, v, k, s, n, n_a, t)
    return ξx(ρ, ϕ, Ξv, s)
end

"""
    ξy(ρ, ϕ, ψ, u, v, k, s, n, n_a, t)

y-polarized integrand with normal-incidence plate (single or stacked).
"""
function ξy(ρ, ϕ, ψ, u, v, k, s, n, n_a, t)
    Ξv = Ξ(ρ, ϕ, ψ, u, v, k, s, n, n_a, t)
    return ξy(ρ, ϕ, Ξv, s)
end

"""
    ξz(ρ, ϕ, ψ, u, v, k, s, n, n_a, t)

z-polarized integrand with normal-incidence plate (single or stacked).
"""
function ξz(ρ, ϕ, ψ, u, v, k, s, n, n_a, t)
    Ξv = Ξ(ρ, ϕ, ψ, u, v, k, s, n, n_a, t)
    return ξz(ρ, ϕ, Ξv, s)
end

"""
    psf_inner(ψ, u, v, k, s, n, n_a, t; rtol=1e-4, atol=1e-5)

Compute the PSF with normal incidence on a plane-parallel plate or a **stack**.

- `n_a` : scalar (single plate) or vector of relative indices n_layer / n_imm
- `t`   : scalar thickness or vector of thicknesses in µm
- `k`   : wave number (2π n_imm / λ)
"""
function psf_inner(ψ, u, v, k, s, n, n_a, t; rtol=1e-4, atol=1e-5)
    resx = hcubature(
        dr -> ξx(dr[1], dr[2], ψ, u, v, k, s, n, n_a, t),
        [0.0, 0.0], [1.0, 2π],
        rtol=rtol, atol=atol
    )
    resy = hcubature(
        dr -> ξy(dr[1], dr[2], ψ, u, v, k, s, n, n_a, t),
        [0.0, 0.0], [1.0, 2π],
        rtol=rtol, atol=atol
    )
    resz = hcubature(
        dr -> ξz(dr[1], dr[2], ψ, u, v, k, s, n, n_a, t),
        [0.0, 0.0], [1.0, 2π],
        rtol=rtol, atol=atol
    )

    ret = 0.0
    ret += abs2(resx[1]::ComplexF64)
    ret += abs2(resy[1]::ComplexF64)
    ret += abs2(resz[1]::ComplexF64)
    return ret / π^2
end

# -------------------------------------------------------------------
# Tilted incidence with a plane-parallel plate or plate stack
# -------------------------------------------------------------------

"""
    Ξ(ρ, ϕ, ψ, u, v, k, s, α, n, n_a, t)

Phase term for tilted incidence with a plane-parallel plate or **stack**.
Uses the tilted aberration function Φ(sρ, ϕ, n_a, t, α).
"""
function Ξ(ρ, ϕ, ψ, u, v, k, s, α, n, n_a, t)
    Φv = Φ(s * ρ, ϕ, n_a, t, α)
    @fastmath Ξv = k * Φv + Ξ(ρ, ϕ, ψ, u, v, s, n)
end

"""
    ξx(ρ, ϕ, ψ, u, v, k, s, α, n, n_a, t)

x-polarized integrand with tilted-incidence plate (single or stacked).
"""
function ξx(ρ, ϕ, ψ, u, v, k, s, α, n, n_a, t)
    Ξv = Ξ(ρ, ϕ, ψ, u, v, k, s, α, n, n_a, t)
    return ξx(ρ, ϕ, Ξv, s)
end

"""
    ξy(ρ, ϕ, ψ, u, v, k, s, α, n, n_a, t)

y-polarized integrand with tilted-incidence plate (single or stacked).
"""
function ξy(ρ, ϕ, ψ, u, v, k, s, α, n, n_a, t)
    Ξv = Ξ(ρ, ϕ, ψ, u, v, k, s, α, n, n_a, t)
    return ξy(ρ, ϕ, Ξv, s)
end

"""
    ξz(ρ, ϕ, ψ, u, v, k, s, α, n, n_a, t)

z-polarized integrand with tilted-incidence plate (single or stacked).
"""
function ξz(ρ, ϕ, ψ, u, v, k, s, α, n, n_a, t)
    Ξv = Ξ(ρ, ϕ, ψ, u, v, k, s, α, n, n_a, t)
    return ξz(ρ, ϕ, Ξv, s)
end

"""
    psf_inner(ψ, u, v, k, s, α, n, n_a, t; rtol=1e-4, atol=1e-5)

Compute the PSF with tilted incidence on a plane-parallel plate or **stack**.

- `α`  : tilt angle (radians)
- `n_a`: scalar or vector of relative indices
- `t`  : scalar or vector of thicknesses (µm)
"""
function psf_inner(ψ, u, v, k, s, α, n, n_a, t; rtol=1e-4, atol=1e-5)
    resx = hcubature(
        dr -> ξx(dr[1], dr[2], ψ, u, v, k, s, α, n, n_a, t),
        [0.0, 0.0], [1.0, 2π],
        rtol=rtol, atol=atol
    )
    resy = hcubature(
        dr -> ξy(dr[1], dr[2], ψ, u, v, k, s, α, n, n_a, t),
        [0.0, 0.0], [1.0, 2π],
        rtol=rtol, atol=atol
    )
    resz = hcubature(
        dr -> ξz(dr[1], dr[2], ψ, u, v, k, s, α, n, n_a, t),
        [0.0, 0.0], [1.0, 2π],
        rtol=rtol, atol=atol
    )

    ret = 0.0
    ret += abs2(resx[1]::ComplexF64)
    ret += abs2(resy[1]::ComplexF64)
    ret += abs2(resz[1]::ComplexF64)
    return ret / π^2
end
# -------------------------------------------------------------------
# High-level PSF(...) methods
# -------------------------------------------------------------------

"""
    PSF(x, y, z, λ, NA; rtol=1e-4, atol=1e-5)

Aberration-free PSF with no objective structure, using air (n=1).  
- `λ` in µm
- `NA` numeric aperture
- `(x, y, z)` in µm
- `rtol`, `atol` tolerances
"""
function PSF(x, y, z, λ, NA; rtol=1e-4, atol=1e-5)
    n = 1.0
    s = NA
    k = 2π / λ
    u = k * (1 - sqrt(1 - s^2)) / (1 - sqrt(1 - s^2 / n^2)) * z
    v = k * s * sqrt(x^2 + y^2)
    ψ = atan(x, y)
    return psf_inner(ψ, u, v, s, n; rtol=rtol, atol=atol)
end

"""
    PSF(x, y, z, λ, obj::Objective; rtol=1e-4, atol=1e-5)

Aberration-free PSF with an `Objective` specifying immersion index, NA, etc.  
No plate, normal incidence in air or immersion.
"""
function PSF(x, y, z, λ, obj::Objective; rtol::Float64=1e-4, atol::Float64=1e-5)
    n_imm = obj.n
    s     = obj.NA / n_imm
    k     = 2π * n_imm / λ

    # s^2/(n^2)? If there's no plate, we consider n=1 in sample side.
    u = k * (1 - sqrt(1 - s^2)) / (1 - sqrt(1 - s^2)) * z
    v = k * s * sqrt(x^2 + y^2)
    ψ = atan(x, y)
    return psf_inner(ψ, u, v, s, 1.0; rtol=rtol, atol=atol)
end

"""
    PSF(x, y, z, λ, NA::Float64, plate::PlaneParallelPlate; ...)

Normal incidence with a plane-parallel plate, no explicit `Objective`.
"""
function PSF(x, y, z, λ, NA::Float64, plate::PlaneParallelPlate; target::PlaneParallelPlate=plate, rtol=1e-4, atol=1e-5)
    n   = target.n_λ(λ)
    n_a = plate.n_λ(λ)
    t   = plate.t
    s   = NA
    k   = 2π / λ
    u   = k * (1 - sqrt(1 - s^2)) / (1 - sqrt(1 - s^2 / n^2)) * z
    v   = k * s * sqrt(x^2 + y^2)
    ψ   = atan(x, y)
    return psf_inner(ψ, u, v, k, s, n, n_a, t; rtol=rtol, atol=atol)
end

"""
    PSF(x, y, z, λ, obj::Objective, plate::PlaneParallelPlate; ...)

Normal incidence with a plane-parallel plate and an explicit Objective.
"""
function PSF(x, y, z, λ, obj::Objective, plate::PlaneParallelPlate; target::PlaneParallelPlate=plate, rtol=1e-4, atol=1e-5)
    n_imm = obj.n
    s     = obj.NA / n_imm
    k     = 2π * n_imm / λ
    n     = target.n_λ(λ)
    n_a   = plate.n_λ(λ) / n_imm
    t     = plate.t
    u = k * (1 - sqrt(1 - s^2)) / (1 - sqrt(1 - s^2 / n^2)) * z
    v = k * s * sqrt(x^2 + y^2)
    ψ = atan(x, y)
    return psf_inner(ψ, u, v, k, s, n, n_a, t; rtol=rtol, atol=atol)
end

"""
    PSF(x, y, z, λ, NA::Float64, plate::PlaneParallelPlate, α; ...)

Tilted incidence with no explicit `Objective`.
"""
function PSF(x, y, z, λ, NA::Float64, plate::PlaneParallelPlate, α; target::PlaneParallelPlate=plate, rtol=1e-4, atol=1e-5)
    n   = target.n_λ(λ)
    n_a = plate.n_λ(λ)
    t   = plate.t
    s   = NA
    k   = 2π / λ
    u   = k * (1 - sqrt(1 - s^2)) / (1 - sqrt(1 - s^2 / n^2)) * z
    v   = k * s * sqrt(x^2 + y^2)
    ψ   = atan(x, y)
    return psf_inner(ψ, u, v, k, s, α, n, n_a, t; rtol=rtol, atol=atol)
end

"""
    PSF(x, y, z, λ, obj::Objective, plate::PlaneParallelPlate, α; ...)

Tilted incidence with plane + objective.
"""
function PSF(x, y, z, λ, obj::Objective, plate::PlaneParallelPlate, α; target::PlaneParallelPlate=plate, rtol=1e-4, atol=1e-5)
    n_imm = obj.n
    s     = obj.NA / n_imm
    k     = 2π * n_imm / λ
    n     = target.n_λ(λ)
    n_a   = plate.n_λ(λ) / n_imm
    t     = plate.t
    u = k * (1 - sqrt(1 - s^2)) / (1 - sqrt(1 - s^2 / n^2)) * z
    v = k * s * sqrt(x^2 + y^2)
    ψ = atan(x, y)
    return psf_inner(ψ, u, v, k, s, α, n, n_a, t; rtol=rtol, atol=atol)
end

# -------------------------------------------------------------------
# Parametric dispatch (optional)
# -------------------------------------------------------------------

"""
    abstract type PSFMode

Abstract type for dispatching different PSF modes 
(e.g. `NoAberration`, `NormalIncidence`, `TiltedIncidence`).
"""
abstract type PSFMode end

"""
    struct NoAberration <: PSFMode end

No plate, no tilt scenario.
"""
struct NoAberration <: PSFMode end

"""
    struct NormalIncidence <: PSFMode end

Normal incidence with a plate scenario.
"""
struct NormalIncidence <: PSFMode end

"""
    struct TiltedIncidence <: PSFMode end

Tilted incidence with a plate scenario.
"""
struct TiltedIncidence <: PSFMode end

"""
    mutable struct PSFParams{M<:PSFMode}

Holds parameters for a certain PSF mode `M`.

- `λ::Float64`: wavelength in µm
- `obj::Objective`: the objective
- `plate`: either a `PlaneParallelPlate` or `nothing`
- `α::Float64`: tilt angle (radians)
"""
mutable struct PSFParams{M<:PSFMode}
    λ::Float64
    obj::Objective
    plate::Union{PlaneParallelPlate,Nothing}
    α::Float64
end

"""
    PSFParams(λ::Float64, obj::Objective)

Construct a `PSFParams{NoAberration}`, i.e. no plate, no tilt.
"""
function PSFParams(λ::Float64, obj::Objective)
    return PSFParams{NoAberration}(λ, obj, nothing, 0.0)
end

"""
    PSF(x, y, z, param::PSFParams{NoAberration}; rtol=1e-4, atol=1e-5)

Dispatch for the no-aberration case (`NoAberration`). 
Calls `PSF(x, y, z, λ, obj; rtol, atol)`.
"""
function PSF(x, y, z, param::PSFParams{NoAberration};
    rtol::Float64=1e-4, atol::Float64=1e-5)
    return PSF(x, y, z, param.λ, param.obj; rtol=rtol, atol=atol)
end

"""
    PSFParams(λ::Float64, obj::Objective, plate::PlaneParallelPlate)

Construct a `PSFParams{NormalIncidence}`, i.e. normal incidence with plate.
"""
function PSFParams(λ::Float64, obj::Objective, plate::PlaneParallelPlate)
    return PSFParams{NormalIncidence}(λ, obj, plate, 0.0)
end

"""
    PSF(x, y, z, param::PSFParams{NormalIncidence}; rtol=1e-4, atol=1e-5)

Dispatch for normal incidence with a plate. 
Calls `PSF(x, y, z, λ, obj, plate; rtol, atol)`.
"""
function PSF(x, y, z, param::PSFParams{NormalIncidence};
    rtol::Float64=1e-4, atol::Float64=1e-5)
    return PSF(x, y, z, param.λ, param.obj, param.plate; rtol=rtol, atol=atol)
end

"""
    PSFParams(λ::Float64, obj::Objective, plate::PlaneParallelPlate, α::Float64)

Construct a `PSFParams{TiltedIncidence}`, i.e. tilted incidence with plate.
`α` in radians.
"""
function PSFParams(λ::Float64, obj::Objective, plate::PlaneParallelPlate, α::Float64)
    return PSFParams{TiltedIncidence}(λ, obj, plate, α)
end

"""
    PSF(x, y, z, param::PSFParams{TiltedIncidence}; rtol=1e-4, atol=1e-5)

Dispatch for tilted incidence with a plate.
Calls `PSF(x, y, z, λ, obj, plate, α; rtol, atol)`.
"""
function PSF(x, y, z, param::PSFParams{TiltedIncidence};
    rtol::Float64=1e-4, atol::Float64=1e-5)
    return PSF(x, y, z, param.λ, param.obj, param.plate, param.α; rtol=rtol, atol=atol)
end

"""
    PSF(x, y, z, λ, NA::Float64, stack::PlateStack; target=last(stack.plates), ...)

Normal incidence with a stack of plane-parallel plates, no explicit `Objective`.

- `stack` : PlateStack of layers between objective and sample.
- `target`: which plate defines the sample refractive index `n` (defaults to last).
"""
function PSF(x, y, z, λ, NA::Float64, stack::PlateStack;
             target::PlaneParallelPlate = last(stack.plates),
             rtol::Float64=1e-4, atol::Float64=1e-5)

    n   = target.n_λ(λ)                         # sample index
    n_a = [p.n_λ(λ) for p in stack.plates]      # absolute plate indices (n_imm = 1)
    t   = [p.t       for p in stack.plates]

    s   = NA
    k   = 2π / λ
    u   = k * (1 - sqrt(1 - s^2)) / (1 - sqrt(1 - s^2 / n^2)) * z
    v   = k * s * sqrt(x^2 + y^2)
    ψ   = atan(x, y)
    return psf_inner(ψ, u, v, k, s, n, n_a, t; rtol=rtol, atol=atol)
end

"""
    PSF(x, y, z, λ, obj::Objective, stack::PlateStack; target=last(stack.plates), ...)

Normal incidence with an Objective and a stack of plane-parallel plates.

- `obj.n` is the immersion index n_imm.
- `n_a` is stored as n_layer / n_imm for each plate.
"""
function PSF(x, y, z, λ, obj::Objective, stack::PlateStack;
             target::PlaneParallelPlate = last(stack.plates),
             rtol::Float64=1e-4, atol::Float64=1e-5)

    n_imm = obj.n
    s     = obj.NA / n_imm
    k     = 2π * n_imm / λ

    n     = target.n_λ(λ)                       # sample index
    n_a   = [p.n_λ(λ) / n_imm for p in stack.plates]
    t     = [p.t             for p in stack.plates]

    u = k * (1 - sqrt(1 - s^2)) / (1 - sqrt(1 - s^2 / n^2)) * z
    v = k * s * sqrt(x^2 + y^2)
    ψ = atan(x, y)
    return psf_inner(ψ, u, v, k, s, n, n_a, t; rtol=rtol, atol=atol)
end

"""
    PSF(x, y, z, λ, NA::Float64, stack::PlateStack, α; ...)

Tilted incidence with a stack, no explicit `Objective`.
"""
function PSF(x, y, z, λ, NA::Float64, stack::PlateStack, α;
             target::PlaneParallelPlate = last(stack.plates),
             rtol::Float64=1e-4, atol::Float64=1e-5)

    n   = target.n_λ(λ)
    n_a = [p.n_λ(λ) for p in stack.plates]
    t   = [p.t       for p in stack.plates]

    s   = NA
    k   = 2π / λ
    u   = k * (1 - sqrt(1 - s^2)) / (1 - sqrt(1 - s^2 / n^2)) * z
    v   = k * s * sqrt(x^2 + y^2)
    ψ   = atan(x, y)
    return psf_inner(ψ, u, v, k, s, α, n, n_a, t; rtol=rtol, atol=atol)
end

"""
    PSF(x, y, z, λ, obj::Objective, stack::PlateStack, α; ...)

Tilted incidence with an Objective and a stack of plates.
"""
function PSF(x, y, z, λ, obj::Objective, stack::PlateStack, α;
             target::PlaneParallelPlate = last(stack.plates),
             rtol::Float64=1e-4, atol::Float64=1e-5)

    n_imm = obj.n
    s     = obj.NA / n_imm
    k     = 2π * n_imm / λ

    n     = target.n_λ(λ)
    n_a   = [p.n_λ(λ) / n_imm for p in stack.plates]
    t     = [p.t             for p in stack.plates]

    u = k * (1 - sqrt(1 - s^2)) / (1 - sqrt(1 - s^2 / n^2)) * z
    v = k * s * sqrt(x^2 + y^2)
    ψ = atan(x, y)
    return psf_inner(ψ, u, v, k, s, α, n, n_a, t; rtol=rtol, atol=atol)
end