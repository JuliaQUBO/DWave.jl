@doc raw"""
    DWaveArchitecture
"""
abstract type DWaveArchitecture <: QUBOTools.AbstractArchitecture end

include("device.jl")
include("topology.jl")
include("chimera.jl")
