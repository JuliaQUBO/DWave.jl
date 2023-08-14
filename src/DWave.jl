module DWave

using PythonCall

import QUBODrivers:
    MOI,
    QUBODrivers,
    QUBOTools,
    Sample,
    SampleSet,
    @setup,
    sample,
    ising

# -*- :: Python D-Wave Module :: -*- #
const np  = PythonCall.pynew()
const plt = PythonCall.pynew()
const dwave_cloud     = PythonCall.pynew()
const dwave_dimod     = PythonCall.pynew()
const dwave_embedding = PythonCall.pynew()
const dwave_networkx  = PythonCall.pynew()
const dwave_system    = PythonCall.pynew()

function __init__()
    # Python Packages
    PythonCall.pycopy!(np, pyimport("numpy"))
    PythonCall.pycopy!(plt, pyimport("matplotlib.pyplot"))
    PythonCall.pycopy!(dwave_cloud, pyimport("dwave.cloud"))
    PythonCall.pycopy!(dwave_dimod, pyimport("dimod"))
    PythonCall.pycopy!(dwave_embedding, pyimport("dwave.embedding"))
    PythonCall.pycopy!(dwave_networkx, pyimport("dwave_networkx"))
    PythonCall.pycopy!(dwave_system, pyimport("dwave.system"))

    # D-Wave API Credentials
    if !haskey(ENV, "DWAVE_API_TOKEN")
        @warn """
        The 'DWAVE_API_TOKEN' environment variable is not defined. Please, make sure that another access method is available.
        
        For more information visit:
            https://docs.ocean.dwavesys.com/en/stable/overview/sapi.html
        """
    end
end

@setup Optimizer begin
    name       = "D-Wave"
    sense      = :min
    domain     = :spin
    version    = v"6.4.1" # dwave-ocean-sdk version
    attributes = begin
        NumberOfReads["num_reads"]::Integer = 100
        Sampler["sampler"]::Any             = nothing
    end
end

function sample(sampler::Optimizer{T}) where {T}
    # Ising Model
    h, J, α, β = ising(sampler, Dict)

    n = MOI.get(sampler, MOI.NumberOfVariables())

    # Attributes
    num_reads     = MOI.get(sampler, DWave.NumberOfReads())
    dwave_sampler = MOI.get(sampler, DWave.Sampler())

    if dwave_sampler === nothing
        dwave_sampler = dwave_system.EmbeddingComposite(
            dwave_system.DWaveSampler(;
                token = get(ENV, "DWAVE_API_TOKEN", nothing)
            )
        )
    end

    # Extra Information
    metadata = Dict{String,Any}(
        "time"   => Dict{String,Any}(),
        "origin" => "D-Wave",
    )

    # Results vector
    samples = Sample{T,Int}[]

    results = @timed dwave_sampler.sample_ising(h, J; num_reads=num_reads)
    var_map = pyconvert.(Int, results.value.variables)

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

        sample = Sample{T,Int}(
            # state:
            ψ,
            # energy:
            α * (pyconvert(T, λ) + β),
            # reads:
            pyconvert(Int, r),
        )

        push!(samples, sample)
    end

    metadata["dwave_info"] = results.value.info

    metadata["time"]["effective"] = results.time

    return SampleSet{T}(samples, metadata)
end

end # module
