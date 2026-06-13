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
    version    = DWave.OCEAN_SDK_VERSION
    attributes = begin
        "num_reads"::Union{Integer,Nothing} = nothing
        "time_limit"::Union{Real,Nothing} = nothing
        "max_num_samples"::Integer = 1_000
        RandomSeed["seed"]::Union{Integer,Nothing} = nothing
    end
end

QUBODrivers.honors_final_reads(::Type{<:Optimizer}) = true
QUBODrivers.enforces_time_limit(::Type{<:Optimizer}) = true

function QUBODrivers.sample(sampler::Optimizer{T}) where {T}
    n, h, J, α, β = QUBOTools.ising(sampler, :dense; sense = :min)

    num_reads = MOI.get(sampler, MOI.RawOptimizerAttribute("num_reads"))
    final_num_reads = MOI.get(sampler, QUBODrivers.FinalNumberOfReads())
    time_limit = MOI.get(sampler, MOI.RawOptimizerAttribute("time_limit"))
    moi_time_limit = MOI.get(sampler, MOI.TimeLimitSec())
    seed = MOI.get(sampler, QUBODrivers.RandomSeed())
    params = Dict{Symbol,Any}(
        :num_reads => final_num_reads,
        :time_limit => isnothing(time_limit) ? moi_time_limit : time_limit,
        :max_num_samples => MOI.get(sampler, MOI.RawOptimizerAttribute("max_num_samples")),
        :seed => seed,
    )

    results = @timed dwave_samplers.RandomSampler().sample_ising(Py(h), Py(J); params...)

    return DWave._format_classical_sampleset(
        T,
        results,
        n,
        α,
        β;
        origin = "D-Wave Random",
        algorithm_name = "D-Wave Random Sampler",
        number_of_reads = num_reads,
        final_number_of_reads = final_num_reads,
        seed = seed,
    )
end

end # module Random
