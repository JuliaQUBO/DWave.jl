module Neal

import ..DWave
import QUBODrivers
import QUBOTools
import MathOptInterface as MOI

using PythonCall

# -*- :: Python D-Wave Simulated Annealing :: -*- #
const np = PythonCall.pynew() # initially NULL
const dwave_samplers = PythonCall.pynew() # initially NULL
"""
Tracks how `dwave_samplers` was initialized.

- `:uninitialized`: `DWave.Neal.__init__()` has not run yet.
- `:narrow`: the wrapper rebuilt a minimal `dwave.samplers.sa` package tree and
  imported only `dwave.samplers.sa.sampler`.
- `:fallback`: Windows fell back to Python's standard import path for
  `dwave.samplers.sa.sampler`.
"""
const dwave_samplers_import_mode = Ref{Symbol}(:uninitialized)

function _clear_sa_import_state!()
    return DWave._clear_dwave_samplers_import_state!()
end

function __init__()
    PythonCall.pycopy!(np, pyimport("numpy"))
    DWave._init_dwave_samplers_target!(dwave_samplers, dwave_samplers_import_mode, "sa.sampler")

    return nothing
end

@doc raw"""
    DWave.Neal.Optimizer

D-Wave's Simulated Annealing Sampler for QUBO and Ising models.
"""
QUBODrivers.@setup Optimizer begin
    name       = "D-Wave Neal Simulated Annealing Sampler"
    version    = v"8.4.0" # dwave-ocean-sdk version
    attributes = begin
        "num_reads"::Integer = 1_000
        "num_sweeps"::Integer = 1_000
        "num_sweeps_per_beta"::Integer = 1
        "beta_range"::Union{Tuple{Float64,Float64},Nothing} = nothing
        "beta_schedule"::Union{Vector,Nothing} = nothing
        "beta_schedule_type"::String = "geometric"
        "seed"::Union{Integer,Nothing} = nothing
        "initial_states_generator"::String = "random"
        "interrupt_function"::Union{Function,Nothing} = nothing
    end
end

function QUBODrivers.sample(sampler::Optimizer{T}) where {T}
    n, h, J, α, β = QUBOTools.ising(sampler, :dense; sense = :min)

    params = Dict{Symbol,Any}(
        :num_reads => MOI.get(sampler, MOI.RawOptimizerAttribute("num_reads")),
        :num_sweeps => MOI.get(sampler, MOI.RawOptimizerAttribute("num_sweeps")),
        :num_sweeps_per_beta => MOI.get(sampler, MOI.RawOptimizerAttribute("num_sweeps_per_beta")),
        :beta_range => MOI.get(sampler, MOI.RawOptimizerAttribute("beta_range")),
        :beta_schedule => MOI.get(sampler, MOI.RawOptimizerAttribute("beta_schedule")),
        :beta_schedule_type => MOI.get(sampler, MOI.RawOptimizerAttribute("beta_schedule_type")),
        :seed => MOI.get(sampler, MOI.RawOptimizerAttribute("seed")),
        :initial_states_generator => MOI.get(sampler, MOI.RawOptimizerAttribute("initial_states_generator")),
        :interrupt_function => MOI.get(sampler, MOI.RawOptimizerAttribute("interrupt_function")),
    )

    py_sampler = dwave_samplers.SimulatedAnnealingSampler()
    results = @timed py_sampler.sample_ising(Py(h), Py(J); params...)

    return DWave._format_classical_sampleset(T, results, n, α, β; origin = "D-Wave Neal")
end

end # module Neal
