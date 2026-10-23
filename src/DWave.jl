module DWave

import Graphs
import JSON
import MathOptInterface as MOI
import QUBOTools
import QUBODrivers

using PythonCall

# -*- :: Python D-Wave Module :: -*- #
const np              = PythonCall.pynew()
const json            = PythonCall.pynew()
const dwave_cloud     = PythonCall.pynew()
const dwave_dimod     = PythonCall.pynew()
const dwave_embedding = PythonCall.pynew()
const dwave_networkx  = PythonCall.pynew()
const dwave_system    = PythonCall.pynew()

const API_TOKEN = Ref{Union{String,Nothing}}(nothing)
const OCEAN_SDK_VERSION = v"9.3.0"
const _OCEAN_BACKEND_NAME = "dwave-ocean-sdk"
const _DWAVE_TIMING_MICROSECOND_KEYS = (
    "qpu_access_time",
    "qpu_sampling_time",
    "qpu_anneal_time_per_sample",
    "qpu_readout_time_per_sample",
    "qpu_delay_time_per_sample",
    "qpu_programming_time",
    "qpu_access_overhead_time",
    "total_post_processing_time",
    "post_processing_overhead_time",
)

const _SAPI_DOCS_URL = "https://docs.ocean.dwavesys.com/en/stable/overview/sapi.html"
const _DWAVE_ENV_VARS = (
    "DWAVE_CONFIG_FILE",
    "DWAVE_PROFILE",
    "DWAVE_API_CLIENT",
    "DWAVE_API_REGION",
    "DWAVE_API_ENDPOINT",
    "DWAVE_API_TOKEN",
    "DWAVE_API_SOLVER",
    "DWAVE_API_PROXY",
    "DWAVE_API_HEADERS",
    "DWAVE_CACHE_ENABLED",
    "DWAVE_CACHE_HOME",
    "DWAVE_LEAP_API_ENDPOINT",
    "DWAVE_LEAP_CLIENT_ID",
    "DWAVE_METADATA_API_ENDPOINT",
)

function _maybe_getproperty(object, name::Symbol)
    try
        return getproperty(object, name)
    catch
        return nothing
    end
end

function _env_api_token()
    token = get(ENV, "DWAVE_API_TOKEN", nothing)
    token === nothing && return nothing
    stripped = strip(token)
    return isempty(stripped) ? nothing : String(stripped)
end

function _sync_dwave_env_to_python!()
    os_mod = try
        pyimport("os")
    catch
        nothing
    end
    os_environ = os_mod === nothing ? nothing : _maybe_getproperty(os_mod, :environ)
    os_environ === nothing && return nothing

    for key in _DWAVE_ENV_VARS
        val = get(ENV, key, nothing)
        if val === nothing
            os_environ.pop(key, pybuiltins.None)
        else
            os_environ[key] = val
        end
    end
    return nothing
end

function _with_synced_dwave_env(f::Function)
    env_token = _env_api_token()
    try
        return withenv("DWAVE_API_TOKEN" => env_token) do
            _sync_dwave_env_to_python!()
            return f(env_token)
        end
    finally
        _sync_dwave_env_to_python!()
    end
end

function _missing_token_message()
    return """
    The 'DWAVE_API_TOKEN' environment variable is not defined or is empty.
    No D-Wave Ocean configuration token was found; if you want to use D-Wave's cloud services, please make sure that another access method is available.

    For more information visit:
        $(_SAPI_DOCS_URL)
    """
end

function _invalid_token_message(err = nothing; from_env::Bool = true)
    detail = err === nothing ? "" : "\nCause: $(sprint(showerror, err))\n"
    header = if from_env
        "The 'DWAVE_API_TOKEN' environment variable defined, but the token is not valid."
    else
        "The configured D-Wave cloud credentials or endpoint are not valid."
    end
    return """
    $(header)$(detail)
    If you want to use D-Wave's cloud services, please make sure that another access method is available.

    For more information visit:
        $(_SAPI_DOCS_URL)
    """
end

function _is_missing_token_error(err)
    details = sprint(showerror, err)
    return occursin("API token not defined", details)
end

function _create_cloud_client(
    token::Union{String, Nothing};
    client_type::String = "qpu",
)
    from_config = _maybe_getproperty(dwave_cloud.Client, :from_config)
    if from_config !== nothing
        defaults = pydict(
            Dict("solver" => pydict(Dict("order_by" => "-num_active_qubits"))),
        )
        return if token === nothing
            from_config(; client = client_type, defaults = defaults)
        else
            from_config(; client = client_type, defaults = defaults, token = token)
        end
    end
    if token === nothing
        error("API token not defined")
    end
    return dwave_cloud.Client(; token = token)
end

function _resolved_client_token(client, fallback::Union{String, Nothing})
    config = _maybe_getproperty(client, :config)
    raw = config === nothing ? _maybe_getproperty(client, :token) : _maybe_getproperty(config, :token)
    if raw !== nothing
        token_str = try
            val = jl_object(raw)
            val === nothing ? "" : strip(String(val))
        catch
            ""
        end
        isempty(token_str) || return token_str
    end
    return fallback
end

function __auth__(; verbose :: Bool = false, throw_error :: Bool = false)
    # D-Wave API Credentials (environment variable or Ocean configuration fallback)
    env_token = _env_api_token()

    try
        _with_synced_dwave_env() do synced_token
            client = _create_cloud_client(synced_token)
            try
                resolved_token = _resolved_client_token(client, synced_token)
                resolved_token === nothing && error("API token not defined")
                client.get_solver()
                API_TOKEN[] = resolved_token
            finally
                close_fn = _maybe_getproperty(client, :close)
                close_fn === nothing || close_fn()
            end
        end

        return !isnothing(API_TOKEN[])
    catch e
        API_TOKEN[] = nothing
        msg = if _is_missing_token_error(e)
            _missing_token_message()
        else
            _invalid_token_message(e; from_env = !isnothing(env_token))
        end

        if verbose
            @warn msg
        end
        if throw_error
            throw(ArgumentError(chomp(msg)))
        end

        return false
    end
end

function __init__()
    # Python Packages
    PythonCall.pycopy!(np, pyimport("numpy"))
    PythonCall.pycopy!(json, pyimport("json"))
    PythonCall.pycopy!(dwave_cloud, pyimport("dwave.cloud"))
    PythonCall.pycopy!(dwave_dimod, pyimport("dimod"))
    PythonCall.pycopy!(dwave_embedding, pyimport("dwave.embedding"))
    PythonCall.pycopy!(dwave_networkx, pyimport("dwave_networkx"))
    PythonCall.pycopy!(dwave_system, pyimport("dwave.system"))

    API_TOKEN[] = nothing

    return nothing
end

function _json_data(value)
    if value isa AbstractDict
        return Dict{String,Any}(string(k) => _json_data(v) for (k, v) in pairs(value))
    elseif value isa AbstractVector
        return Any[_json_data(v) for v in value]
    else
        return value
    end
end

function jl_object(py_obj)
    # Convert Python object to JSON string, then parse it into a Julia object
    data = pyconvert(String, json.dumps(py_obj))

    return _json_data(JSON.parse(data))
end

function _metadata_base(;
    origin::String,
    algorithm_name::String,
    execution_mode::String,
    number_of_reads::Integer,
    final_number_of_reads::Integer = number_of_reads,
    optimizer_iterations = nothing,
    optimizer_evaluations = final_number_of_reads,
    seeds::Dict{String,Any} = Dict{String,Any}(),
)
    return Dict{String,Any}(
        "origin" => origin,
        "algorithm" => Dict{String,Any}(
            "name" => algorithm_name,
        ),
        "backend" => Dict{String,Any}(
            "name" => _OCEAN_BACKEND_NAME,
            "version" => OCEAN_SDK_VERSION,
        ),
        "execution" => Dict{String,Any}(
            "mode" => execution_mode,
        ),
        "optimizer" => Dict{String,Any}(
            "iterations" => optimizer_iterations,
            "evaluations" => optimizer_evaluations,
        ),
        "reads" => Dict{String,Any}(
            "number_of_reads" => number_of_reads,
            "final_number_of_reads" => final_number_of_reads,
        ),
        "seeds" => seeds,
        "status" => "locally_solved",
        "termination_status" => MOI.LOCALLY_SOLVED,
    )
end

function _seed_metadata(seed)
    return isnothing(seed) ? Dict{String,Any}() : Dict{String,Any}("sampler" => seed)
end

function _dwave_timing(dwave_info::AbstractDict)
    timing = get(dwave_info, "timing", nothing)

    timing isa AbstractDict || return Dict{String,Any}()

    return Dict{String,Any}(string(k) => v for (k, v) in pairs(timing))
end

function _dwave_timing_seconds(value)
    value isa Real || return nothing

    return value / 1_000_000
end

function _dwave_effective_time(dwave_info::AbstractDict, fallback::Real)
    timing = _dwave_timing(dwave_info)
    access_time = get(timing, "qpu_access_time", nothing)
    effective = _dwave_timing_seconds(access_time)

    return isnothing(effective) ? fallback : effective
end

function _attach_dwave_timing!(metadata::Dict{String,Any}, dwave_info::AbstractDict)
    timing = _dwave_timing(dwave_info)

    isempty(timing) && return metadata

    metadata["time"]["dwave"] = Dict{String,Any}(
        "timing" => timing,
        "units" => Dict{String,Any}(
            key => "microseconds" for key in keys(timing) if key in _DWAVE_TIMING_MICROSECOND_KEYS
        ),
    )

    return metadata
end

include("wrapper/architecture.jl")

include("sampler.jl")
include("classical.jl")
include("neal/sampler.jl")
include("greedy/sampler.jl")
include("random/sampler.jl")
include("tabu/sampler.jl")

end # module
