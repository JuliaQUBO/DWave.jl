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
    version    = DWave.OCEAN_SDK_VERSION
    attributes = begin
        "num_reads"::Integer = 1_000
        "num_sweeps"::Integer = 1_000
        "num_sweeps_per_beta"::Integer = 1
        "beta_range"::Union{Tuple{Float64,Float64},Nothing} = nothing
        "beta_schedule"::Union{Vector,Nothing} = nothing
        "beta_schedule_type"::String = "geometric"
        RandomSeed["seed"]::Union{Integer,Nothing} = nothing
        "initial_states_generator"::String = "random"
        "interrupt_function"::Union{Function,Nothing} = nothing
    end
end

QUBODrivers.honors_final_reads(::Type{<:Optimizer}) = true

function _sparse_ising_bqm(n::Int, h::Dict{Int,T}, J::Dict{Tuple{Int,Int},T}) where {T}
    linear = zeros(T, n)

    for (i, v) in h
        linear[i] = v
    end

    quadratic = collect(J)
    # Match dimod's dense-matrix interaction order so seeded Neal runs stay stable.
    sort!(quadratic; by = term -> (first(term)[2], first(term)[1]))

    rows = Vector{Int}(undef, length(quadratic))
    cols = Vector{Int}(undef, length(quadratic))
    weights = Vector{T}(undef, length(quadratic))

    for k in eachindex(quadratic)
        (i, j) = first(quadratic[k])

        rows[k] = j - 1
        cols[k] = i - 1
        weights[k] = last(quadratic[k])
    end

    return DWave.dwave_dimod.BinaryQuadraticModel.from_numpy_vectors(
        np.array(linear),
        (np.array(rows), np.array(cols), np.array(weights)),
        zero(T),
        DWave.dwave_dimod.SPIN,
    )
end

function QUBODrivers.sample(sampler::Optimizer{T}) where {T}
    n, h, J, α, β = QUBOTools.ising(sampler, :dict; sense = :min)

    num_reads = MOI.get(sampler, MOI.RawOptimizerAttribute("num_reads"))
    final_num_reads = MOI.get(sampler, QUBODrivers.FinalNumberOfReads())
    seed = MOI.get(sampler, QUBODrivers.RandomSeed())
    params = Dict{Symbol,Any}(
        :num_reads => final_num_reads,
        :num_sweeps => MOI.get(sampler, MOI.RawOptimizerAttribute("num_sweeps")),
        :num_sweeps_per_beta => MOI.get(sampler, MOI.RawOptimizerAttribute("num_sweeps_per_beta")),
        :beta_range => MOI.get(sampler, MOI.RawOptimizerAttribute("beta_range")),
        :beta_schedule => MOI.get(sampler, MOI.RawOptimizerAttribute("beta_schedule")),
        :beta_schedule_type => MOI.get(sampler, MOI.RawOptimizerAttribute("beta_schedule_type")),
        :seed => seed,
        :initial_states_generator => MOI.get(
            sampler,
            MOI.RawOptimizerAttribute("initial_states_generator"),
        ),
        :interrupt_function => MOI.get(sampler, MOI.RawOptimizerAttribute("interrupt_function")),
    )

    py_sampler = dwave_samplers.SimulatedAnnealingSampler()
    results = @timed py_sampler.sample(_sparse_ising_bqm(n, h, J); params...)

    return DWave._format_classical_sampleset(
        T,
        results,
        n,
        α,
        β;
        origin = "D-Wave Neal",
        algorithm_name = "D-Wave Neal Simulated Annealing Sampler",
        number_of_reads = num_reads,
        final_number_of_reads = final_num_reads,
        seed = seed,
    )
end

end # module Neal
