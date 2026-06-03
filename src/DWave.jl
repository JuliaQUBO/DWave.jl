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

function jl_object(py_obj)
    # Convert Python object to JSON string, then parse it into a Julia object
    data = pyconvert(String, json.dumps(py_obj))

    return JSON.parse(data)
end

include("sampler.jl")
include("classical.jl")
include("neal/sampler.jl")
include("greedy/sampler.jl")
include("random/sampler.jl")
include("tabu/sampler.jl")

end # module
