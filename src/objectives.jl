# objectives.jl
# Struct definitions and constructors for microscope objectives.

"""
    Objective

A microscope objective characterized by:
- `f::Float64`: focal length (µm)
- `NA::Float64`: numerical aperture
- `n::Float64`: refractive index of immersion medium (e.g., 1.0 for air)
"""
struct Objective
    f::Float64    # Focal length in micrometers
    NA::Float64   # Numerical aperture (dimensionless)
    n::Float64    # Refractive index in the immersion medium
end
Objective(f::Float64, NA::Float64) = Objective(f::Float64, NA::Float64, 1.0) 
"""
    MPlanApo100x()

Returns an `Objective` resembling the Mitsutoyo MPlanApo 100x:
- `f=2000` µm
- `NA=0.7`
- `n=1.0` (air)
"""
function MPlanApo100x()
    return Objective(2000, 0.7, 1.0)
end

"""
    MPlanApo50x()

Returns an `Objective` resembling the Mitsutoyo MPlanApo 50x:
- `f=4000` µm
- `NA=0.55`
- `n=1.0` (air)
"""
function MPlanApo50x()
    return Objective(4000, 0.55, 1.0)
end

"""
    LMPLFLN100XBD()

Returns an `Objective` resembling the Olympus LMPLFLN 100x BD:
- `f=1800` µm
- `NA=0.8`
- `n=1.0` (air)
"""
function LMPLFLN100XBD()
    return Objective(1800, 0.8, 1.0)
end

"""
    M10x()

Returns an `Objective` resembling the Newport M10x:
- `f=16500` µm
- `NA=0.25`
- `n=1.0` (air)
"""
function M10x()
    return Objective(16500, 0.25, 1.0)
end

"""
    UPLXAPO100X()

Returns an `Objective` resembling the Olympus UPLXAPO100X:
- `f=1800` µm
- `NA=1.45`
- `n=1.515` (oil immersion)
"""
function UPLXAPO100X()
    return Objective(1800, 1.45, 1.515)
end
