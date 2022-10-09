module DWave

using Anneal
using PythonCall

# -*- :: Python D-Wave Module :: -*- #
const dwave_cloud = PythonCall.pynew()

function __init__()
    PythonCall.pycopy!(dwave_cloud, pyimport("dwave.cloud"))
end

Anneal.@anew Optimizer begin
    name = "D-Wave"
    sense = :min
    domain = :spin
    version = v"0.1.0"
    attributes = begin
        "num_reads"::Integer = 100
    end
end

function Anneal.sample(sampler::Optimizer{T}) where {T}
    # Ising Model
    J, h = Anneal.ising(Dict, T, sampler)

    # Attributes
    num_reads = MOI.get(sampler, MOI.RawOptimizerAttribute("num_reads"))

    # Results vector
    samples = Anneal.Sample{Int,T}[]

    # Time data
    time_data = Dict{String,Any}()

    connect() do client
        solver = client.get_solver()
        result = @timed solver.sample_ising(J, h; num_reads=num_reads)
        record = result.value.record

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

end # module
