# plane_parallel_plates.jl
# This file contains definitions for plane-parallel plates and their Sellmeier-based refractive index curves.

"""
    PlaneParallelPlate

A `PlaneParallelPlate` has thickness `t` (in micrometers) and a refractive index function `n_λ(λ)`,
where `λ` is in micrometers. The field `n_λ` returns the refractive index at that wavelength.
"""
struct PlaneParallelPlate
    t::Float64         # Thickness in micrometers
    n_λ::Function      # Refractive index curve, n(λ) for λ in µm
end

"""
    Platestack

A `Platestack` is a vector of PlaneParallelPlate to handle the case one has multiple different media in the optical path.
"""
struct PlateStack
    plates::Vector{PlaneParallelPlate}
end
PlateStack(ps::PlaneParallelPlate...) = PlateStack(collect(ps))
# ------------------------------------------------------------
# Diamond
# ------------------------------------------------------------

"""
    Diamond(t)

Approximate dispersion for diamond with thickness `t` (in µm).
`λ` is assumed in micrometers (µm), but the formula internally converts λ to nm 
by multiplying by 1e3 for the Sellmeier-like expression.
"""
function Diamond(t)
    n_λ(λ) = sqrt(
        1 + (4.658 * (λ * 1e3)^2) / ((λ * 1e3)^2 - 112.5^2)
    )
    return PlaneParallelPlate(Float64(t), n_λ)
end

# ------------------------------------------------------------
# Sellmeier Utility
# ------------------------------------------------------------

"""
    sellmeier(λ, B, C)

Generic Sellmeier equation:

```
n^2 = 1 + ∑( Bᵢ * λ² / (λ² - Cᵢ) )
```

- `λ` in micrometers (µm).
- `B`/`C` are matching-length vectors of Sellmeier coefficients.

Returns the refractive index `n(λ)`.
"""
function sellmeier(λ, B::AbstractVector{<:Real}, C::AbstractVector{<:Real})
    @assert length(B) == length(C) "B and C must have the same length."
    λ2 = λ^2
    val = 0.0
    @inbounds @simd for i in eachindex(B)
        val += B[i] * λ2 / (λ2 - C[i])
    end
    return sqrt(1 + val)
end

# ------------------------------------------------------------
# Fused Silica
# ------------------------------------------------------------

"""
    FusedSilica(t)

Creates a `PlaneParallelPlate` of thickness `t` (µm) for Fused Silica (SiO₂),
using the Malitson Sellmeier coefficients:
```
B = [0.6961663, 0.4079426, 0.8974794]
C = [0.0684043^2, 0.1162414^2, 9.896161^2]
```
Source: [Malitson data](https://refractiveindex.info/?shelf=main&book=SiO2&page=Malitson).
"""
function FusedSilica(t)
    B = [0.6961663, 0.4079426, 0.8974794]
    C = [0.0684043^2, 0.1162414^2, 9.896161^2]

    n_λ(λ) = sellmeier(λ, B, C)
    return PlaneParallelPlate(t, n_λ)
end

# ------------------------------------------------------------
# Borosilicate Crown (BK7)
# ------------------------------------------------------------

"""
    BorosilicateCrown(t)

Creates a `PlaneParallelPlate` of thickness `t` (µm) for BK7 (borosilicate crown glass).
Uses the Schott Sellmeier coefficients:

```
B = [1.040, 0.230, 1.010]
C = [0.006, 0.020, 103.560]
```
Source: [refractiveindex.info BK7 page](https://refractiveindex.info/?shelf=glass&book=BK7&page=SCHOTT).
"""
function BorosilicateCrown(t)
    B = [1.040, 0.230, 1.010]
    C = [0.006, 0.020, 103.560]

    n_λ(λ) = sellmeier(λ, B, C)
    return PlaneParallelPlate(t, n_λ)
end

# ------------------------------------------------------------
# Sapphire (ordinary ray)
# ------------------------------------------------------------

"""
    Sapphire(t)

Creates a `PlaneParallelPlate` of thickness `t` (µm) for sapphire (Al₂O₃),
using the ordinary-ray Sellmeier coefficients (Li):
```
B = [1.4313493, 0.65054713, 5.3414021]
C = [0.0726631^2, 0.1193242^2, 18.028251^2]
```
Source: [Sapphire (Li) data](https://refractiveindex.info/?shelf=main&book=Al2O3&page=Li).
"""
function Sapphire(t)
    B = [1.4313493, 0.65054713, 5.3414021]
    C = [0.0726631^2, 0.1193242^2, 18.028251^2]

    n_λ(λ) = sellmeier(λ, B, C)
    return PlaneParallelPlate(t, n_λ)
end

# ------------------------------------------------------------
# Magnesium Fluoride (MgF₂) ordinary ray
# ------------------------------------------------------------

"""
    MagnesiumFluoride(t)

Creates a `PlaneParallelPlate` of thickness `t` (µm) for MgF₂ 
using the ordinary-ray Sellmeier coefficients (Li):
```
B = [0.48755108, 0.39875031, 2.3120353]
C = [0.04338408^2, 0.09461442^2, 23.793604^2]
```
Source: [MgF₂ (Li) data](https://refractiveindex.info/?shelf=main&book=MgF2&page=Li).
"""
function MagnesiumFluoride(t)
    B = [0.48755108, 0.39875031, 2.3120353]
    C = [0.04338408^2, 0.09461442^2, 23.793604^2]

    n_λ(λ) = sellmeier(λ, B, C)
    return PlaneParallelPlate(t, n_λ)
end

# ------------------------------------------------------------
# Custom Plate
# ------------------------------------------------------------

"""
    CustomPlate(t, B, C)

Create a `PlaneParallelPlate` of thickness `t` (µm), with a refractive index 
defined by the Sellmeier coefficients `B` and `C`, where
```
n^2 = 1 + ∑( Bᵢ * λ² / (λ² - Cᵢ) )
```
(λ in micrometers). 
`B` and `C` must be the same length.
"""
function CustomPlate(t::Float64, B::AbstractVector{<:Real}, C::AbstractVector{<:Real})
    @assert length(B) == length(C) "B and C must have the same length."
    n_λ(λ) = sellmeier(λ, B, C)
    return PlaneParallelPlate(t, n_λ)
end