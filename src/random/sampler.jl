module Random

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
    DWave._init_dwave_samplers_target!(dwave_samplers, dwave_samplers_import_mode, "random.sampler")

    return nothing
end

@doc raw"""
    DWave.Random.Optimizer

D-Wave's random sampler for QUBO and Ising models.
"""
QUBODrivers.@setup Optimizer begin
    name       = "D-Wave Random Sampler"
    version    = v"9.3.0" # dwave-ocean-sdk version
    attributes = begin
        "num_reads"::Union{Integer,Nothing} = nothing
        "time_limit"::Union{Real,Nothing} = nothing
        "max_num_samples"::Integer = 1_000
        "seed"::Union{Integer,Nothing} = nothing
    end
end

function QUBODrivers.sample(sampler::Optimizer{T}) where {T}
    n, h, J, α, β = QUBOTools.ising(sampler, :dense; sense = :min)

    params = Dict{Symbol,Any}(
        :num_reads => MOI.get(sampler, MOI.RawOptimizerAttribute("num_reads")),
        :time_limit => MOI.get(sampler, MOI.RawOptimizerAttribute("time_limit")),
        :max_num_samples => MOI.get(sampler, MOI.RawOptimizerAttribute("max_num_samples")),
        :seed => MOI.get(sampler, MOI.RawOptimizerAttribute("seed")),
    )

    results = @timed dwave_samplers.RandomSampler().sample_ising(Py(h), Py(J); params...)

    return DWave._format_classical_sampleset(T, results, n, α, β; origin = "D-Wave Random")
end

end # module Random
