module Tabu

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
    DWave._init_dwave_samplers_target!(
        dwave_samplers,
        dwave_samplers_import_mode,
        "tabu";
        leaf_is_package = true,
    )

    return nothing
end

@doc raw"""
    DWave.Tabu.Optimizer

D-Wave's tabu-search sampler for QUBO and Ising models.
"""
QUBODrivers.@setup Optimizer begin
    name       = "D-Wave Tabu Sampler"
    version    = v"8.4.0" # dwave-ocean-sdk version
    attributes = begin
        "initial_states"::Any = nothing
        "initial_states_generator"::String = "random"
        "num_reads"::Union{Integer,Nothing} = nothing
        "seed"::Union{Integer,Nothing} = nothing
        "tenure"::Union{Integer,Nothing} = nothing
        "timeout"::Union{Integer,Nothing} = 20
        "num_restarts"::Integer = 1_000_000
        "energy_threshold"::Any = nothing
        "coefficient_z_first"::Union{Integer,Nothing} = nothing
        "coefficient_z_restart"::Union{Integer,Nothing} = nothing
        "lower_bound_z"::Union{Integer,Nothing} = nothing
    end
end

function QUBODrivers.sample(sampler::Optimizer{T}) where {T}
    n, h, J, α, β = QUBOTools.ising(sampler, :dense; sense = :min)

    params = Dict{Symbol,Any}(
        :initial_states => DWave._normalize_initial_states(
            n,
            MOI.get(sampler, MOI.RawOptimizerAttribute("initial_states")),
        ),
        :initial_states_generator => MOI.get(
            sampler,
            MOI.RawOptimizerAttribute("initial_states_generator"),
        ),
        :num_reads => MOI.get(sampler, MOI.RawOptimizerAttribute("num_reads")),
        :seed => MOI.get(sampler, MOI.RawOptimizerAttribute("seed")),
        :tenure => MOI.get(sampler, MOI.RawOptimizerAttribute("tenure")),
        :timeout => MOI.get(sampler, MOI.RawOptimizerAttribute("timeout")),
        :num_restarts => MOI.get(sampler, MOI.RawOptimizerAttribute("num_restarts")),
        :energy_threshold => MOI.get(sampler, MOI.RawOptimizerAttribute("energy_threshold")),
        :coefficient_z_first => MOI.get(
            sampler,
            MOI.RawOptimizerAttribute("coefficient_z_first"),
        ),
        :coefficient_z_restart => MOI.get(
            sampler,
            MOI.RawOptimizerAttribute("coefficient_z_restart"),
        ),
        :lower_bound_z => MOI.get(sampler, MOI.RawOptimizerAttribute("lower_bound_z")),
    )

    results = @timed dwave_samplers.TabuSampler().sample_ising(Py(h), Py(J); params...)

    return DWave._format_classical_sampleset(T, results, n, α, β; origin = "D-Wave Tabu")
end

end # module Tabu
