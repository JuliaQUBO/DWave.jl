module Neal

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
"""
const dwave_samplers_import_mode = Ref{Symbol}(:uninitialized)

"""
Clear the cached `dwave.samplers` module tree before rebuilding the Neal import state.

This mutates Python's `sys.modules` and should only be used during package
initialization or in tests that need a clean import environment.
"""
function _clear_sa_import_state!()
    PythonCall.pyexec(
        """
for name in tuple(sys.modules):
    if name == "dwave.samplers" or name.startswith("dwave.samplers."):
        sys.modules.pop(name, None)

if hasattr(dwave, "samplers"):
    del dwave.samplers
""",
        @__MODULE__,
        (
            dwave = pyimport("dwave"),
            sys = pyimport("sys"),
        ),
    )

    return nothing
end

function _import_sa_sampler()
    locals = (
        dwave = pyimport("dwave"),
        importlib = pyimport("importlib"),
        pathlib = pyimport("pathlib"),
        sys = pyimport("sys"),
    )

    ans = PythonCall.pyexec(
        @NamedTuple{sampler::PythonCall.Py},
        """
root = next(iter(dwave.__path__), None)
if root is None:
    raise ImportError("dwave package does not define an import path")

root = pathlib.Path(root)
samplers_name = "dwave.samplers"
sa_name = "dwave.samplers.sa"
sampler_name = "dwave.samplers.sa.sampler"
samplers_dir = root / "samplers"
sa_dir = samplers_dir / "sa"

samplers_spec = importlib.util.spec_from_file_location(
    samplers_name,
    samplers_dir / "__init__.py",
    submodule_search_locations=[str(samplers_dir)],
)
samplers_pkg = importlib.util.module_from_spec(samplers_spec)
# Keep the real package search path, but do not execute __init__ because that
# eagerly imports unrelated samplers such as dwave.samplers.random.
sys.modules[samplers_name] = samplers_pkg

dwave.samplers = samplers_pkg

sa_spec = importlib.util.spec_from_file_location(
    sa_name,
    sa_dir / "__init__.py",
    submodule_search_locations=[str(sa_dir)],
)
sa_pkg = importlib.util.module_from_spec(sa_spec)
# As above, keep a valid package object for submodule resolution without running
# dwave.samplers.sa.__init__.
sys.modules[sa_name] = sa_pkg

samplers_pkg.sa = sa_pkg

spec = importlib.util.spec_from_file_location(
    sampler_name,
    sa_dir / "sampler.py",
)
sampler = importlib.util.module_from_spec(spec)
sys.modules[sampler_name] = sampler
spec.loader.exec_module(sampler)

sa_pkg.sampler = sampler
""",
        @__MODULE__,
        locals,
    )

    return ans.sampler
end

function __init__()
    PythonCall.pycopy!(np, pyimport("numpy"))
    # Note: 'neal' package was deprecated and replaced by 'dwave.samplers' in
    # dwave-ocean-sdk 8.0+. Load the simulated annealing submodule directly so
    # we do not execute dwave.samplers.__init__ and pull in unrelated samplers.
    # Rebuild the package tree from a clean Python import state so repeated
    # initialization and partially imported modules do not leak into the wrapper.
    _clear_sa_import_state!()
    PythonCall.pycopy!(dwave_samplers, _import_sa_sampler())
    dwave_samplers_import_mode[] = :narrow

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
