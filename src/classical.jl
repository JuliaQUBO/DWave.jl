"""
Clear the cached `dwave.samplers` module tree.

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

function _dwave_samplers_import_name(target::String)
    return "dwave.samplers.$target"
end

"""
Clear only the cached subtree for a specific `dwave.samplers` target.

This preserves unrelated sampler subtrees that were already imported through
other wrappers.
"""
function _clear_dwave_samplers_import_target!(target::String)
    PythonCall.pyexec(
        """
samplers_name = "dwave.samplers"
target_name = target_import_name

prefixes = {target_name}
target_parts = target_name.split(".")[2:]
for i in range(1, len(target_parts)):
    prefixes.add(f"{samplers_name}." + ".".join(target_parts[:i]))

names_to_remove = set()
for module_name in tuple(sys.modules):
    for prefix in prefixes:
        if module_name == prefix or module_name.startswith(prefix + "."):
            names_to_remove.add(module_name)
            break

for name in sorted(names_to_remove, key=lambda item: item.count("."), reverse=True):
    mod = sys.modules.pop(name, None)
    if mod is None:
        continue

    parent_name, _, child_name = name.rpartition(".")
    if parent_name and parent_name in sys.modules:
        parent_mod = sys.modules[parent_name]
        if getattr(parent_mod, child_name, None) is mod:
            delattr(parent_mod, child_name)

if samplers_name in sys.modules:
    dwave.samplers = sys.modules[samplers_name]
elif hasattr(dwave, "samplers"):
    del dwave.samplers
""",
        @__MODULE__,
        (
            dwave = pyimport("dwave"),
            sys = pyimport("sys"),
            target_import_name = _dwave_samplers_import_name(target),
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
        importlib_util = pyimport("importlib.util"),
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

def build_spec_or_raise(name, location, submodule_search_locations=None, importlib_util=importlib_util):
    if not location.exists():
        raise ImportError(f"Could not build module spec for {name} from {location}")
    spec = importlib_util.spec_from_file_location(
        name,
        location,
        submodule_search_locations=submodule_search_locations,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not build module spec for {name} from {location}")
    return spec

def ensure_package_stub(
    name,
    directory,
    parent_pkg=None,
    attr_name=None,
    sys=sys,
    importlib_util=importlib_util,
    build_spec_or_raise=build_spec_or_raise,
):
    expected_path = [str(directory)]
    pkg_mod = sys.modules.get(name)
    if pkg_mod is None or list(getattr(pkg_mod, "__path__", [])) != expected_path:
        pkg_spec = build_spec_or_raise(
            name,
            directory / "__init__.py",
            submodule_search_locations=expected_path,
        )
        pkg_mod = importlib_util.module_from_spec(pkg_spec)
        # Register a package stub with the real search path, but do not execute
        # __init__ because that would eagerly import unrelated samplers.
        sys.modules[name] = pkg_mod

    if parent_pkg is not None and attr_name is not None:
        setattr(parent_pkg, attr_name, pkg_mod)

    return pkg_mod

samplers_pkg = ensure_package_stub(samplers_name, samplers_dir)
dwave.samplers = samplers_pkg

parts = target.split(".")
parent_name = samplers_name
parent_pkg = samplers_pkg
parent_dir = samplers_dir

for part in parts[:-1]:
    parent_dir = parent_dir / part
    pkg_name = f"{parent_name}.{part}"
    pkg_mod = ensure_package_stub(pkg_name, parent_dir, parent_pkg, part)
    parent_pkg = pkg_mod
    parent_name = pkg_name

leaf = parts[-1]
leaf_name = f"{parent_name}.{leaf}"

if leaf_is_package:
    leaf_dir = parent_dir / leaf
    leaf_spec = build_spec_or_raise(
        leaf_name,
        leaf_dir / "__init__.py",
        submodule_search_locations=[str(leaf_dir)],
    )
    leaf_mod = importlib_util.module_from_spec(leaf_spec)
    sys.modules[leaf_name] = leaf_mod
    setattr(parent_pkg, leaf, leaf_mod)
    leaf_spec.loader.exec_module(leaf_mod)
else:
    leaf_spec = build_spec_or_raise(
        leaf_name,
        parent_dir / f"{leaf}.py",
    )
    leaf_mod = importlib_util.module_from_spec(leaf_spec)
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
    # Keep sibling sampler modules alive in sys.modules by clearing only the
    # target subtree, prefer the narrow import path that avoids executing
    # dwave.samplers.__init__, and only fall back to Python's standard import
    # machinery on Windows when compiled extensions require it. That fallback
    # must start from a fully clean dwave.samplers tree so Python can rebuild
    # the real package hierarchy instead of reusing our synthetic root stub.
    _clear_dwave_samplers_import_target!(target)

    try
        PythonCall.pycopy!(target_ref, _import_dwave_samplers_target(target; leaf_is_package))
        import_mode[] = :narrow
    catch err
        if Sys.iswindows()
            _clear_dwave_samplers_import_state!()
            PythonCall.pycopy!(target_ref, pyimport(_dwave_samplers_import_name(target)))
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

function _format_classical_sampleset(
    ::Type{T},
    results,
    n::Int,
    α,
    β;
    origin::String,
    algorithm_name::String,
    number_of_reads = nothing,
    final_number_of_reads = number_of_reads,
    seed = nothing,
) where {T}
    samples = QUBOTools.Sample{T,Int}[]
    var_map = pyconvert.(Int, [var for var in results.value.variables]) .+ 1
    observed_reads = 0

    for row in results.value.record
        ψ = zeros(Int, n)

        for (i, v) in enumerate(row["sample"])
            ψ[var_map[i]] = pyconvert(Int, v)
        end

        reads = pyconvert(Int, row["num_occurrences"])
        observed_reads += reads

        push!(
            samples,
            QUBOTools.Sample{T,Int}(
                ψ,
                α * (pyconvert(T, row["energy"]) + β),
                reads,
            ),
        )
    end

    dwave_info = jl_object(results.value.info)
    metadata_number_of_reads = something(number_of_reads, observed_reads)
    metadata_final_number_of_reads = something(final_number_of_reads, observed_reads)
    metadata = _metadata_base(
        origin = origin,
        algorithm_name = algorithm_name,
        execution_mode = "local",
        number_of_reads = metadata_number_of_reads,
        final_number_of_reads = metadata_final_number_of_reads,
        seeds = _seed_metadata(seed),
    )
    metadata["time"] = Dict{String,Any}("effective" => results.time)
    _attach_dwave_timing!(metadata, dwave_info)
    metadata["dwave_info"] = dwave_info

    return QUBOTools.SampleSet{T,Int}(samples, metadata; sense = :min, domain = :spin)
end
