"""
Clear the cached `dwave.samplers` module tree before rebuilding a classical
sampler import state.

This mutates Python's `sys.modules` and should only be used during package
initialization or in tests that need a clean import environment.
"""
function _clear_dwave_samplers_import_state!()
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

"""
Import a target below `dwave.samplers` without executing
`dwave.samplers.__init__`.

When `leaf_is_package = true`, the target is treated as a package relative to
`dwave.samplers`; otherwise the target is treated as a module path.
"""
function _import_dwave_samplers_target(target::String; leaf_is_package::Bool = false)
    locals = (
        dwave = pyimport("dwave"),
        importlib = pyimport("importlib"),
        pathlib = pyimport("pathlib"),
        sys = pyimport("sys"),
        target = target,
        leaf_is_package = leaf_is_package,
    )

    ans = PythonCall.pyexec(
        @NamedTuple{target_module::PythonCall.Py},
        """
root = next(iter(dwave.__path__), None)
if root is None:
    raise ImportError("dwave package does not define an import path")

root = pathlib.Path(root)
samplers_name = "dwave.samplers"
samplers_dir = root / "samplers"

samplers_spec = importlib.util.spec_from_file_location(
    samplers_name,
    samplers_dir / "__init__.py",
    submodule_search_locations=[str(samplers_dir)],
)
samplers_pkg = importlib.util.module_from_spec(samplers_spec)
sys.modules[samplers_name] = samplers_pkg
dwave.samplers = samplers_pkg

parts = target.split(".")
parent_name = samplers_name
parent_pkg = samplers_pkg
parent_dir = samplers_dir

for part in parts[:-1]:
    parent_dir = parent_dir / part
    pkg_name = f"{parent_name}.{part}"
    pkg_spec = importlib.util.spec_from_file_location(
        pkg_name,
        parent_dir / "__init__.py",
        submodule_search_locations=[str(parent_dir)],
    )
    pkg_mod = importlib.util.module_from_spec(pkg_spec)
    sys.modules[pkg_name] = pkg_mod
    setattr(parent_pkg, part, pkg_mod)
    parent_pkg = pkg_mod
    parent_name = pkg_name

leaf = parts[-1]
leaf_name = f"{parent_name}.{leaf}"

if leaf_is_package:
    leaf_dir = parent_dir / leaf
    leaf_spec = importlib.util.spec_from_file_location(
        leaf_name,
        leaf_dir / "__init__.py",
        submodule_search_locations=[str(leaf_dir)],
    )
    leaf_mod = importlib.util.module_from_spec(leaf_spec)
    sys.modules[leaf_name] = leaf_mod
    setattr(parent_pkg, leaf, leaf_mod)
    leaf_spec.loader.exec_module(leaf_mod)
else:
    leaf_spec = importlib.util.spec_from_file_location(
        leaf_name,
        parent_dir / f"{leaf}.py",
    )
    leaf_mod = importlib.util.module_from_spec(leaf_spec)
    sys.modules[leaf_name] = leaf_mod
    setattr(parent_pkg, leaf, leaf_mod)
    leaf_spec.loader.exec_module(leaf_mod)

target_module = leaf_mod
""",
        @__MODULE__,
        locals,
    )

    return ans.target_module
end

"""
Initialize a classical `dwave.samplers` target in a `pynew()` slot.

On Windows this falls back to Python's standard import machinery if the narrow
import path fails, which is required for some compiled extensions.
"""
function _init_dwave_samplers_target!(
    target_ref::PythonCall.Py,
    import_mode::Base.RefValue{Symbol},
    target::String;
    leaf_is_package::Bool = false,
)
    _clear_dwave_samplers_import_state!()

    try
        PythonCall.pycopy!(target_ref, _import_dwave_samplers_target(target; leaf_is_package))
        import_mode[] = :narrow
    catch err
        if Sys.iswindows()
            _clear_dwave_samplers_import_state!()
            PythonCall.pycopy!(target_ref, pyimport("dwave.samplers.$target"))
            import_mode[] = :fallback
        else
            rethrow(err)
        end
    end

    return nothing
end

function _normalize_initial_states(n::Int, initial_states)
    if initial_states === nothing || initial_states isa PythonCall.Py
        return initial_states
    elseif initial_states isa AbstractVector
        length(initial_states) == n || throw(
            ArgumentError("`initial_states` must have length $n, got $(length(initial_states))"),
        )

        return Py((reshape(Int.(collect(initial_states)), 1, n), collect(0:(n - 1))))
    elseif initial_states isa AbstractMatrix
        size(initial_states, 2) == n || throw(
            ArgumentError(
                "`initial_states` must have $n columns, got $(size(initial_states, 2))",
            ),
        )

        return Py((Int.(Matrix(initial_states)), collect(0:(n - 1))))
    else
        return Py(initial_states)
    end
end

function _format_classical_sampleset(::Type{T}, results, n::Int, α, β; origin::String) where {T}
    samples = QUBOTools.Sample{T,Int}[]
    var_map = pyconvert.(Int, [var for var in results.value.variables]) .+ 1

    for row in results.value.record
        ψ = zeros(Int, n)

        for (i, v) in enumerate(row["sample"])
            ψ[var_map[i]] = pyconvert(Int, v)
        end

        push!(
            samples,
            QUBOTools.Sample{T,Int}(
                ψ,
                α * (pyconvert(T, row["energy"]) + β),
                pyconvert(Int, row["num_occurrences"]),
            ),
        )
    end

    metadata = Dict{String,Any}(
        "origin" => origin,
        "time" => Dict{String,Any}(
            "effective" => results.time,
        ),
        "dwave_info" => jl_object(results.value.info),
    )

    return QUBOTools.SampleSet{T,Int}(samples, metadata; sense = :min, domain = :spin)
end
