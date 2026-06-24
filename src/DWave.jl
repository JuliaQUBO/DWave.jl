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

function __auth__(; verbose :: Bool = false)
    # D-Wave API Credentials
    token = get(ENV, "DWAVE_API_TOKEN", nothing)

    if token !== nothing && !isempty(strip(token))
        API_TOKEN[] = token

        try
            client = dwave_cloud.Client(; token = API_TOKEN[])
            client.get_solver()
        catch e
            API_TOKEN[] = nothing

            if verbose
                @warn """
                The 'DWAVE_API_TOKEN' environment variable defined, but the token is not valid.
                If you want to use D-Wave's cloud services, please make sure that another access method is available.
                
                For more information visit:
                    https://docs.ocean.dwavesys.com/en/stable/overview/sapi.html
                """
            end
        end

        return !isnothing(API_TOKEN[])
    else
        API_TOKEN[] = nothing

        if verbose
            @warn """
            The 'DWAVE_API_TOKEN' environment variable is not defined or is empty.
            If you want to use D-Wave's cloud services, please make sure that another access method is available.
            
            For more information visit:
                https://docs.ocean.dwavesys.com/en/stable/overview/sapi.html
            """
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

    __auth__(; verbose = true)

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
