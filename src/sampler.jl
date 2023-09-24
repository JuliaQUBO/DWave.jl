@doc raw"""
    DWave.Optimizer

D-Wave's Quantum Annealing Sampler for QUBO and Ising models.
"""
QUBODrivers.@setup Optimizer begin
    name       = "D-Wave Quantum Annealing Sampler"
    version    = v"6.4.1" # dwave-ocean-sdk version
    attributes = begin
        NumberOfReads["num_reads"]::Integer = 100
        Sampler["sampler"]::Any             = nothing
    end
end

function QUBODrivers.sample(sampler::Optimizer{T}) where {T}
    # Ising Model
    n, h, J, α, β = QUBOTools.ising(sampler, :dict; sense = :min)

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

    # Results vector
    samples = QUBOTools.Sample{T,Int}[]

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

        sample = QUBOTools.Sample{T,Int}(
            # state:
            ψ,
            # energy:
            α * (pyconvert(T, λ) + β),
            # reads:
            pyconvert(Int, r),
        )

        push!(samples, sample)
    end

    # Metadata
    metadata = Dict{String,Any}(
        "origin" => "D-Wave",
        "time"   => Dict{String,Any}( #
            "effective" => results.time,
        ),
        "dwave_info" => results.value.info
    )

    return QUBOTools.SampleSet{T,Int}(samples, metadata; sense = :min, domain = :spin)
end
