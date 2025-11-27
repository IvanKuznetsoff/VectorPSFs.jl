# VectorPSFs.jl
# Main module for vectorial PSF calculations, pulling in submodules for NV spectrum,
# objectives, plane-parallel plates, aberration phase functions, PSF core logic, and 
# Strehl ratio calculations.

"""
    module VectorPSFs

The primary module for computing vector PSFs (point spread functions) with or without
aberrations caused by plane-parallel plates. Also supports NV-center weighted PSFs,
Strehl ratio calculations, and lens objectives.
"""
module VectorPSFs

using HCubature
using StaticArrays
using LinearAlgebra
using Statistics
using SmoothingSplines
using Optim

"""
Include submodules:
- `NVspectrum.jl`: NV spectrum data and spline fitting
- `objectives.jl`: definitions of microscope objectives
- `plane_parallel_plates.jl`: Sellmeier-based materials, diamond, etc.
- `aberration_functions.jl`: phase functions (Φ) for normal/tilted incidence
- `psf_core.jl`: core integrators for PSF
- `nvcenter.jl` (optional): NVCenter weighting
- `strehl.jl`: Strehl ratio computations
"""

include("NVspectrum.jl")         # NVspectrum submodule
include("objectives.jl")         # objective lenses
include("plane_parallel_plates.jl")
include("aberration_functions.jl")
include("psf_core.jl")
include("nvcenter.jl")           # optional NV center submodule
include("strehl.jl")

"""
Exports the public API: structures, methods, and constants from submodules:
- From objectives.jl: `Objective`, standard lens constructors
- From plane_parallel_plates.jl: `PlaneParallelPlate`, materials (Diamond, etc.)
- From NVspectrum.jl: `NVspectrum`, `nv_spectrum_spline`
- From psf_core.jl: `PSF`, `PSFParams`
- From strehl.jl: `strehl`, `maxtol_thick`
"""
# From `objectives.jl`
export Objective
export MPlanApo100x, LMPLFLN100XBD, MPlanApo50x, M10x, UPLXAPO100X

# From `plane_parallel_plates.jl`
export PlaneParallelPlate
export Diamond, FusedSilica, BorosilicateCrown, Sapphire, MagnesiumFluoride
export CustomPlate, PlateStack

# From `NVspectrum.jl`
export NVspectrum
export nv_spectrum_spline

# From `nvcenter.jl`
export NVCenter  

# From `psf_core.jl`
export PSF, PSFParams

# From `strehl.jl`
export strehl, maxtol_thick

end # module VectorPsfs
