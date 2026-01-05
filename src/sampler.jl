@doc raw"""
    DWave.Optimizer

D-Wave's Quantum Annealing Sampler for QUBO and Ising models.
"""
QUBODrivers.@setup Optimizer begin
    name       = "D-Wave Quantum Annealing Sampler"
    version    = v"8.4.0" # dwave-ocean-sdk version
    attributes = begin
        NumberOfReads["num_reads"]::Integer       = 100
        Sampler["sampler"]::Any                   = nothing
        ReturnEmbedding["return_embedding"]::Bool = false
        AnnealingTime["annealing_time"]::Float64  = 20.0
    end
end

function QUBODrivers.sample(sampler::Optimizer{T}) where {T}
    # Ising Model
    n, h, J, α, β = QUBOTools.ising(sampler, :dict; sense = :min)

    # Attributes
    sample_params = Dict{Symbol,Any}(
        :num_reads      => MOI.get(sampler, DWave.NumberOfReads()),
        :annealing_time => MOI.get(sampler, DWave.AnnealingTime()),
    )
    dwave_sampler = MOI.get(sampler, DWave.Sampler())

    if dwave_sampler === nothing
        dwave_sampler = dwave_system.EmbeddingComposite(
            dwave_system.DWaveSampler(; token = get(ENV, "DWAVE_API_TOKEN", nothing)),
        )

        sample_params[:return_embedding] = MOI.get(sampler, DWave.ReturnEmbedding())
    end

    # Results
    samples = QUBOTools.Sample{T,Int}[]
    results = @timed dwave_sampler.sample_ising(h, J; sample_params...)
    var_map = pyconvert.(Int, [var for var in results.value.variables])
    dw_info = jl_object(results.value.info)

    # Extract chip information from the sampler properties
    # When using EmbeddingComposite, access the child DWaveSampler
    chip_info = Dict{String,Any}()
    try
        # Get the underlying DWaveSampler (child of EmbeddingComposite)
        base_sampler =
            hasproperty(dwave_sampler, :child) ? dwave_sampler.child : dwave_sampler

        if hasproperty(base_sampler, :properties)
            props = base_sampler.properties

            # Extract chip_id if available
            if haskey(props, "chip_id")
                chip_info["chip_id"] = pyconvert(String, props["chip_id"])
            end

            # Extract topology information if available
            if haskey(props, "topology")
                topology_info = jl_object(props["topology"])
                chip_info["topology"] = topology_info
            end

            # Extract solver name which often contains model info (e.g., Advantage_system4.1)
            if haskey(props, "solver_name")
                chip_info["solver_name"] = pyconvert(String, props["solver_name"])
            end

            # Extract category (qpu vs software)
            if haskey(props, "category")
                chip_info["category"] = pyconvert(String, props["category"])
            end
        end
    catch e
        # If we can't extract chip info, just continue without it
        # This ensures backward compatibility
    end

    # Add chip info to dw_info if we successfully extracted any
    if !isempty(chip_info)
        dw_info["chip_info"] = chip_info
    end

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
        "time" => Dict{String,Any}( #
            "effective" => results.time,
        ),
        "dwave_info" => dw_info,
    )

    return QUBOTools.SampleSet{T,Int}(samples, metadata; sense = :min, domain = :spin)
end
