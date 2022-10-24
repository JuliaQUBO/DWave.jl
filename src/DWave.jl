module DWave

using Anneal
using PythonCall

# -*- :: Python D-Wave Module :: -*- #
const dwave_dimod = PythonCall.pynew()
const dwave_cloud = PythonCall.pynew()
const dwave_embedding = PythonCall.pynew()

function __init__()
    PythonCall.pycopy!(dwave_dimod, pyimport("dimod"))
    PythonCall.pycopy!(dwave_cloud, pyimport("dwave.cloud"))
    PythonCall.pycopy!(dwave_embedding, pyimport("dwave.embedding"))
end

Anneal.@anew Optimizer begin
    name       = "D-Wave"
    sense      = :min
    domain     = :spin
    version    = v"0.1.0"
    attributes = begin
        "num_reads"::Integer = 100
    end
end

function Anneal.sample(sampler::Optimizer{T}) where {T}
    # Ising Model
    h, J, α, β = Anneal.ising(Dict, T, sampler)

    # D-Wave's BQM
    bqm = DWave.dwave_dimod.BinaryQuadraticModel.from_ising(h, J)

    # Attributes
    num_reads = MOI.get(sampler, MOI.RawOptimizerAttribute("num_reads"))

    # -*- Timing Information -*- #
    time_data = Dict{String,Any}()

    # Results vector
    samples = Anneal.Sample{Int,T}[]

    connect() do client
        solver = client.get_solver()
        
        χ, hχ, Jχ = embed(solver, h, J)

        result = @timed solver.sample_ising(hχ, Jχ; num_reads=num_reads)
        future = result.value
        record = unembed(future.sampleset, χ, bqm).record

        for (ψ, e, k) in record
            sample = Anneal.Sample{Int,T}(
                # state:
                pyconvert.(Int, ψ),
                # reads:
                pyconvert(Int, k),
                # value: 
                α * (pyconvert(T, e) + β),
            )

            push!(samples, sample)
        end

        time_data["effective"] = result.time

        return nothing
    end

    metadata = Dict{String,Any}(
        "time" => time_data,
        "origin" => "D-Wave",
    )

    return Anneal.SampleSet{Int,T}(samples, metadata)
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
end

function embed(solver, h::Dict{Int,T}, J::Dict{Tuple{Int,Int},T}) where {T}
    χ = DWave.dwave_embedding.minorminer.find_embedding(
        keys(J),
        solver.edges
    )

    A = Dict{Int, Vector{Int}}(v => Int[] for v in pyconvert.(Int, solver.nodes))

    for e in solver.edges
        u, v = pyconvert.(Int, e)
        push!(A[u], v)
        push!(A[v], u)
    end

    hχ, Jχ = DWave.dwave_embedding.embed_ising(h, J, χ, A)

    return (χ, hχ, Jχ)
end

function unembed(sampleset, χ, bqm)
    return DWave.dwave_embedding.unembed_sampleset(sampleset, χ, bqm)
end

end # module
