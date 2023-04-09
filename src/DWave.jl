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
const dwave_dimod = PythonCall.pynew()
const dwave_cloud = PythonCall.pynew()
const dwave_embedding = PythonCall.pynew()

function __init__()
    # Python Packages
    PythonCall.pycopy!(dwave_dimod, pyimport("dimod"))
    PythonCall.pycopy!(dwave_cloud, pyimport("dwave.cloud"))
    PythonCall.pycopy!(dwave_embedding, pyimport("dwave.embedding"))

    # D-Wave API Credentials
    DWAVE_API_TOKEN = get(ENV, "DWAVE_API_TOKEN", nothing)

    if isnothing(DWAVE_API_TOKEN)
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
    version    = v"6.0.0" # dwave-ocean-sdk version
    attributes = begin
        NumberOfReads["num_reads"]::Integer   = 100
        DWaveBackend["dwave_backend"]::String = "DW_2000Q_6"
    end
end

function sample(sampler::Optimizer{T}) where {T}
    # Ising Model
    h, J, α, β = ising(sampler, Dict, T)

    # D-Wave's BQM
    bqm = dwave_dimod.BinaryQuadraticModel.from_ising(h, J)

    # Attributes
    num_reads     = MOI.get(sampler, DWave.NumberOfReads())
    dwave_backend = MOI.get(sampler, DWave.DWaveBackend())

    # Extra Information
    metadata = Dict{String,Any}(
        "time"   => Dict{String,Any}(),
        "origin" => "D-Wave @ $(dwave_backend)",
    )

    # Results vector
    samples = Sample{T,Int}[]

    connect() do client
        solver = client.get_solver(dwave_backend)
        
        χ, hχ, Jχ = embed(solver, h, J)

        result = @timed solver.sample_ising(hχ, Jχ; num_reads=num_reads)
        future = result.value
        record = unembed(future.sampleset, χ, bqm).record

        for (ψ, e, k) in record
            sample = Sample{T}(
                # state:
                pyconvert.(Int, ψ),
                # value: 
                α * (pyconvert(T, e) + β),
                # reads:
                pyconvert(Int, k),
            )

            push!(samples, sample)
        end

        metadata["time"]["effective"] = result.time

        return nothing
    end

    return SampleSet{T}(samples, metadata)
end

function connect(callback)
    client = dwave_cloud.Client.from_config()

    try
        callback(client)
    catch e
        rethrow(e)
    finally
        client.close()
    end

    return nothing
end

function embed(solver, h::Dict{Int,T}, J::Dict{Tuple{Int,Int},T}) where {T}
    H = keys(J)
    G = solver.edges
    V = pyconvert.(Int, solver.nodes)

    χ = dwave_embedding.minorminer.find_embedding(H, G)

    A = Dict{Int, Vector{Int}}(v => Int[] for v in V)

    for g in G
        u, v = pyconvert.(Int, g)

        push!(A[u], v)
        push!(A[v], u)
    end

    hχ, Jχ = dwave_embedding.embed_ising(h, J, χ, A)

    return (χ, hχ, Jχ)
end

function unembed(sampleset, χ, bqm)
    return DWave.dwave_embedding.unembed_sampleset(sampleset, χ, bqm)
end

end # module
