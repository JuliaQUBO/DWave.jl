module DWave

import Graphs
import QUBOTools
import QUBODrivers
import MathOptInterface as MOI

using PythonCall

# -*- :: Python D-Wave Module :: -*- #
const np              = PythonCall.pynew()
const dwave_cloud     = PythonCall.pynew()
const dwave_dimod     = PythonCall.pynew()
const dwave_embedding = PythonCall.pynew()
const dwave_networkx  = PythonCall.pynew()
const dwave_system    = PythonCall.pynew()

function __auth__()
    # D-Wave API Credentials
    if !haskey(ENV, "DWAVE_API_TOKEN")
        @warn """
        The 'DWAVE_API_TOKEN' environment variable is not defined.
        If you want to use D-Wave's cloud services, please make sure that another access method is available.
        
        For more information visit:
            https://docs.ocean.dwavesys.com/en/stable/overview/sapi.html
        """
    end

    return nothing
end

function __init__()
    # Python Packages
    PythonCall.pycopy!(np, pyimport("numpy"))
    PythonCall.pycopy!(dwave_cloud, pyimport("dwave.cloud"))
    PythonCall.pycopy!(dwave_dimod, pyimport("dimod"))
    PythonCall.pycopy!(dwave_embedding, pyimport("dwave.embedding"))
    PythonCall.pycopy!(dwave_networkx, pyimport("dwave_networkx"))
    PythonCall.pycopy!(dwave_system, pyimport("dwave.system"))

    __auth__()

    return nothing
end

include("sampler.jl")
include("neal/sampler.jl")

end # module
