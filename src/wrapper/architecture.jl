@doc raw"""
    DWaveArchitecture
"""
abstract type DWaveArchitecture <: QUBOTools.AbstractArchitecture end

include("device.jl")
include("chimera.jl")
include("hfs/format.jl")
