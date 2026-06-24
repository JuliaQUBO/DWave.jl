@doc raw"""
    DWaveArchitecture
"""
abstract type DWaveArchitecture <: QUBOTools.AbstractArchitecture end

include("device.jl")
include("topology.jl")
include("chimera.jl")
# The HFS writer is legacy and depends on QUBOTools format internals that are
# not loaded into DWave's module namespace. Track revive-vs-remove in #48.
