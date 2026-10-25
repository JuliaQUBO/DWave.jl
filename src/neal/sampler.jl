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
        "initial_states"::Any = nothing
        "initial_states_generator"::String = "random"
        "interrupt_function"::Union{Function,Nothing} = nothing
    end
end

const _TIME_LIMIT_CLOCK = Ref{Function}(time_ns)

QUBODrivers.honors_final_reads(::Type{<:Optimizer}) = true

MOI.supports(
    ::Optimizer,
    ::Union{MOI.TimeLimitSec,QUBODrivers.@raw_attr_str("moi/timelimitsec")},
) = true

function MOI.set(
    sampler::Optimizer,
    attr::QUBODrivers.@raw_attr_str("moi/timelimitsec"),
    value::Real,
)
    if value isa Bool || !isfinite(value) || value < zero(value)
        throw(
            ArgumentError(
                "Value for 'MOI.TimeLimitSec' must be a finite non-negative number, or 'nothing'",
            ),
        )
    end

    QUBODrivers.set_raw_attr!(sampler, attr, Float64(value))

    return nothing
end

function MOI.get(sampler::Optimizer{T}, attr::MOI.TerminationStatus) where {T}
    metadata = QUBOTools.metadata(QUBOTools.solution(sampler))
    return get(metadata, "termination_status") do
        invoke(MOI.get, Tuple{QUBODrivers.AbstractSampler{T},MOI.TerminationStatus}, sampler, attr)
    end
end

function MOI.optimize!(sampler::Optimizer{T}) where {T}
    # A failed new solve must not expose the previous solve's result or status.
    QUBOTools.attach!(sampler, QUBOTools.SampleSet{T,Int}())
    try
        return invoke(MOI.optimize!, Tuple{QUBODrivers.AbstractSampler{T}}, sampler)
    catch
        QUBOTools.attach!(sampler, QUBOTools.SampleSet{T,Int}(;
            metadata = Dict{String,Any}(
                "status" => "sampling_error",
                "termination_status" => MOI.OTHER_ERROR,
            ),
        ))
        rethrow()
    end
end

function _tracked_interrupt_function(
    callback::Union{Function,Nothing};
    time_limit_sec::Union{Float64,Nothing} = nothing,
    start_ns::Union{UInt64,Nothing} = nothing,
    clock::Function = _TIME_LIMIT_CLOCK[],
)
    deadline_check = if isnothing(time_limit_sec) || isnothing(start_ns)
        nothing
    else
        () -> Float64(UInt64(clock()) - start_ns) * 1.0e-9 >= time_limit_sec
    end

    return pyexec(
        @NamedTuple{callback::Py,state::Py},
        """
def track_interrupt(original, deadline_check, time_limit_sec):
    state = {
        "interrupt_requested": False,
        "interrupt_callback_error": False,
        "time_limit_exceeded": False,
        "time_limit_sec": time_limit_sec,
        "granularity": "completed_read",
    }

    def tracked(*args, **kwargs):
        user_stop = False
        if original is not None:
            try:
                # Convert exactly once, as the native callback bridge does.
                user_stop = bool(original(*args, **kwargs))
            except Exception:
                state["interrupt_callback_error"] = True
                raise
            if user_stop:
                state["interrupt_requested"] = True

        deadline_stop = False
        if deadline_check is not None:
            deadline_stop = bool(deadline_check())
            if deadline_stop:
                state["time_limit_exceeded"] = True

        return user_stop or deadline_stop

    return tracked, state

callback, state = track_interrupt(original, deadline_check, time_limit_sec)
""",
        @__MODULE__,
        (
            original = callback,
            deadline_check = deadline_check,
            time_limit_sec = time_limit_sec,
        ),
    )
end

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

# Normalize scale in the original domain, before Bool-to-Spin additions.
# This matches MOI's coefficient arithmetic; scaling converted fields can perturb
# an exact floating-point tie and change Neal's automatic cold-beta selection.
function _native_ising_form(sampler::Optimizer{T}) where {T}
    if isone(QUBOTools.scale(sampler))
        return QUBOTools.ising(sampler, :dict; sense = :min)
    end

    n, L, Q, α, β = QUBOTools.form(sampler, :dict)
    # form(..., :dict) owns fresh, sized dictionaries. Preserve their layout:
    # rebuilding through filtered generators can reorder floating-point sums.
    map!(v -> α * v, values(L))
    map!(v -> α * v, values(Q))
    # Drop zero-scaled coefficients as MOI does, retaining all n variables.
    filter!(term -> !iszero(last(term)), L)
    filter!(term -> !iszero(last(term)), Q)
    scaled = QUBOTools.Form{T}(n,
        QUBOTools.DictLinearForm{T}(L), QUBOTools.DictQuadraticForm{T}(Q), one(T), α * β;
        sense = QUBOTools.sense(sampler), domain = QUBOTools.domain(sampler))

    return QUBOTools.cast(QUBOTools.frame(sampler) => QUBOTools.Frame(:min, :spin), scaled)
end

function QUBODrivers.sample(sampler::Optimizer{T}) where {T}
    conversion = @timed begin
        n, h, J, _, β = _native_ising_form(sampler)
        bqm = _sparse_ising_bqm(n, h, J)
        initial_states = DWave._normalize_initial_states(
            n,
            MOI.get(sampler, MOI.RawOptimizerAttribute("initial_states"));
            domain = QUBOTools.domain(sampler),
        )
        (n = n, β = β, bqm = bqm, initial_states = initial_states)
    end
    n = conversion.value.n
    β = conversion.value.β
    bqm = conversion.value.bqm
    initial_states = conversion.value.initial_states

    num_reads = MOI.get(sampler, MOI.RawOptimizerAttribute("num_reads"))
    final_num_reads = MOI.get(sampler, QUBODrivers.FinalNumberOfReads())
    seed = MOI.get(sampler, QUBODrivers.RandomSeed())
    interrupt_function = MOI.get(sampler, MOI.RawOptimizerAttribute("interrupt_function"))
    time_limit_sec = MOI.get(sampler, MOI.TimeLimitSec())
    clock = _TIME_LIMIT_CLOCK[]
    sample_start_ns = UInt64(clock())
    tracked = if isnothing(interrupt_function) && isnothing(time_limit_sec)
        nothing
    else
        _tracked_interrupt_function(
            interrupt_function;
            time_limit_sec = time_limit_sec,
            start_ns = sample_start_ns,
            clock = clock,
        )
    end
    params = Dict{Symbol,Any}(
        :num_reads => final_num_reads,
        :num_sweeps => MOI.get(sampler, MOI.RawOptimizerAttribute("num_sweeps")),
        :num_sweeps_per_beta => MOI.get(sampler, MOI.RawOptimizerAttribute("num_sweeps_per_beta")),
        :beta_range => MOI.get(sampler, MOI.RawOptimizerAttribute("beta_range")),
        :beta_schedule => MOI.get(sampler, MOI.RawOptimizerAttribute("beta_schedule")),
        :beta_schedule_type => MOI.get(sampler, MOI.RawOptimizerAttribute("beta_schedule_type")),
        :seed => seed,
        :initial_states => initial_states,
        :initial_states_generator => MOI.get(
            sampler,
            MOI.RawOptimizerAttribute("initial_states_generator"),
        ),
        :interrupt_function => isnothing(tracked) ? nothing : tracked.callback,
    )

    py_sampler = dwave_samplers.SimulatedAnnealingSampler()
    # The native form is already scaled and in minimization energy units.
    # Constants do not affect transitions; restore only its offset afterward.
    results = @timed py_sampler.sample(bqm; params...)

    sampleset = DWave._format_classical_sampleset(
        T,
        results,
        n,
        one(T),
        β;
        origin = "D-Wave Neal",
        algorithm_name = "D-Wave Neal Simulated Annealing Sampler",
        number_of_reads = num_reads,
        final_number_of_reads = final_num_reads,
        seed = seed,
        report_completed_reads = true,
    )

    metadata = QUBOTools.metadata(sampleset)
    metadata["time"]["conversion"] = conversion.time
    termination = isnothing(tracked) ? Dict{String,Any}(
        "interrupt_requested" => false,
        "interrupt_callback_error" => false,
        "time_limit_exceeded" => false,
        "time_limit_sec" => nothing,
        "granularity" => "completed_read",
    ) : DWave.jl_object(tracked.state)
    completed_reads = metadata["reads"]["completed_number_of_reads"]
    status, reason = if termination["interrupt_callback_error"]
        MOI.OTHER_ERROR, "interrupt_callback_error"
    elseif termination["interrupt_requested"]
        MOI.INTERRUPTED, "interrupt_callback"
    elseif termination["time_limit_exceeded"]
        MOI.TIME_LIMIT, "time_limit"
    elseif completed_reads != final_num_reads || isempty(sampleset)
        MOI.OTHER_ERROR, "incomplete_return"
    else
        MOI.LOCALLY_SOLVED, "completed"
    end
    termination["reason"] = reason
    metadata["termination"] = termination
    metadata["termination_status"] = status
    metadata["status"] = status == MOI.LOCALLY_SOLVED ? "locally_solved" : reason

    return sampleset
end

end # module Neal
