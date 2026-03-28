module Neal

import QUBODrivers
import QUBOTools
import MathOptInterface as MOI

using PythonCall

# -*- :: Python D-Wave Simulated Annealing :: -*- #
const np = PythonCall.pynew() # initially NULL
const dwave_samplers = PythonCall.pynew() # initially NULL

function _import_sa_sampler()
    locals = (
        dwave = pyimport("dwave"),
        importlib = pyimport("importlib"),
        pathlib = pyimport("pathlib"),
        sys = pyimport("sys"),
        types = pyimport("types"),
    )

    ans = PythonCall.pyexec(
        @NamedTuple{sampler::PythonCall.Py},
        """
root = pathlib.Path(next(iter(dwave.__path__)))
samplers_name = "dwave.samplers"
sa_name = "dwave.samplers.sa"

samplers_pkg = sys.modules.get(samplers_name)
if samplers_pkg is None:
    samplers_pkg = types.ModuleType(samplers_name)
    samplers_pkg.__path__ = [str(root / "samplers")]
    samplers_pkg.__package__ = samplers_name
    sys.modules[samplers_name] = samplers_pkg

dwave.samplers = samplers_pkg

sa_pkg = sys.modules.get(sa_name)
if sa_pkg is None:
    sa_pkg = types.ModuleType(sa_name)
    sa_pkg.__path__ = [str(root / "samplers" / "sa")]
    sa_pkg.__package__ = sa_name
    sys.modules[sa_name] = sa_pkg

samplers_pkg.sa = sa_pkg
sampler = importlib.import_module("dwave.samplers.sa.sampler")
""",
        @__MODULE__,
        locals,
    )

    return ans.sampler
end

function __init__()
    PythonCall.pycopy!(np, pyimport("numpy"))
    # Note: 'neal' package was deprecated and replaced by 'dwave.samplers' in
    # dwave-ocean-sdk 8.0+. Construct the intermediate package objects
    # manually so we can import only the simulated annealing module without
    # executing dwave.samplers.__init__ and pulling in unrelated samplers.
    PythonCall.pycopy!(dwave_samplers, _import_sa_sampler())
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
    # Retrieve Ising Model
    n, h, J, α, β = QUBOTools.ising(sampler, :dense; sense = :min)

    # Retrieve Optimizer Attributes
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

    # Call D-Wave Neal API
    sampler = dwave_samplers.SimulatedAnnealingSampler()
    results = @timed sampler.sample_ising(Py(h), Py(J); params...)

    # Format Samples
    samples = QUBOTools.Sample{T,Int}[]
    # NumPy-array Ising inputs label variables 0:(n-1) on the Python side.
    var_map = pyconvert.(Int, [var for var in results.value.variables]) .+ 1

    for (ϕ, λ, r) in results.value.record
        # the dwave sampler will not consider variables that are not
        # present in the objective funcion, leading to holes with
        # respect to the indices in the record table.
        # Therefore, it is necessary to introduce an extra layer of
        # indirection to account for the missing variables.
        ψ = zeros(Int, n)

        for (i, v) in enumerate(ϕ)
            ψ[var_map[i]] = pyconvert(Int, v)
        end

        sample = QUBOTools.Sample{T,Int}(
            # state:
            ψ,
            # energy:
            α * (pyconvert(T, λ) + β),
            # reads:
            pyconvert(Int, r),
        )

        push!(samples, sample)
    end

    # Write metadata
    metadata = Dict{String,Any}(
        "origin" => "D-Wave Neal",
        "time"   => Dict{String,Any}( #
            "effective" => results.time
        ),
    )

    return QUBOTools.SampleSet{T,Int}(samples, metadata; sense = :min, domain = :spin)
end

end # module Neal
