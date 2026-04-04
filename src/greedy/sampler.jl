module Greedy

import ..DWave
import QUBODrivers
import QUBOTools
import MathOptInterface as MOI

using PythonCall

const dwave_samplers = PythonCall.pynew()
const dwave_samplers_import_mode = Ref{Symbol}(:uninitialized)

function _clear_import_state!()
    return DWave._clear_dwave_samplers_import_state!()
end

function __init__()
    DWave._init_dwave_samplers_target!(dwave_samplers, dwave_samplers_import_mode, "greedy.sampler")

    return nothing
end

@doc raw"""
    DWave.Greedy.Optimizer

D-Wave's steepest-descent sampler for QUBO and Ising models.
"""
QUBODrivers.@setup Optimizer begin
    name       = "D-Wave Greedy Steepest Descent Sampler"
    version    = v"8.4.0" # dwave-ocean-sdk version
    attributes = begin
        # Delegate `nothing` to the upstream sampler, which infers num_reads
        # from initial_states or defaults to a single read.
        "num_reads"::Union{Integer,Nothing} = nothing
        "initial_states"::Any = nothing
        "initial_states_generator"::String = "random"
        "seed"::Union{Integer,Nothing} = nothing
        "large_sparse_opt"::Bool = false
    end
end

function QUBODrivers.sample(sampler::Optimizer{T}) where {T}
    n, h, J, α, β = QUBOTools.ising(sampler, :dense; sense = :min)

    params = Dict{Symbol,Any}(
        :num_reads => MOI.get(sampler, MOI.RawOptimizerAttribute("num_reads")),
        :initial_states => DWave._normalize_initial_states(
            n,
            MOI.get(sampler, MOI.RawOptimizerAttribute("initial_states")),
        ),
        :initial_states_generator => MOI.get(
            sampler,
            MOI.RawOptimizerAttribute("initial_states_generator"),
        ),
        :seed => MOI.get(sampler, MOI.RawOptimizerAttribute("seed")),
        :large_sparse_opt => MOI.get(sampler, MOI.RawOptimizerAttribute("large_sparse_opt")),
    )

    results = @timed dwave_samplers.SteepestDescentSampler().sample_ising(Py(h), Py(J); params...)

    return DWave._format_classical_sampleset(T, results, n, α, β; origin = "D-Wave Greedy")
end

end # module Greedy
