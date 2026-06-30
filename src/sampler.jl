@doc raw"""
    DWave.Optimizer

D-Wave's Quantum Annealing Sampler for QUBO and Ising models.
"""
QUBODrivers.@setup Optimizer begin
    name       = "D-Wave Quantum Annealing Sampler"
    version    = DWave.OCEAN_SDK_VERSION
    attributes = begin
        NumberOfReads["num_reads"]::Integer       = 100
        Sampler["sampler"]::Any                   = nothing
        ReturnEmbedding["return_embedding"]::Bool = false
        AnnealingTime["annealing_time"]::Float64  = 20.0
    end
end

QUBODrivers.honors_final_reads(::Type{<:Optimizer}) = true

const _DWAVE_CHIP_INFO_KEYS = (
    "chip_id",
    "topology",
    "solver_name",
    "category",
    # These calibrated working-graph lists are large on current QPUs, but
    # preserving them lets WorkingGraph(metadata) reconstruct the active solver.
    "qubits",
    "couplers",
    "num_qubits",
)

function _maybe_getproperty(object, name::Symbol)
    try
        return getproperty(object, name)
    catch
        return nothing
    end
end

function _as_julia_metadata(value)
    value === nothing && return nothing

    if value isa AbstractString ||
       value isa Number ||
       value isa Bool ||
       value isa AbstractDict ||
       value isa AbstractVector
        return value
    end

    try
        return jl_object(value)
    catch
        return nothing
    end
end

function _metadata_property(properties, key::String)
    if properties isa AbstractDict
        haskey(properties, key) || return nothing

        return _as_julia_metadata(properties[key])
    end

    try
        pyconvert(Bool, properties.__contains__(key)) || return nothing

        return _as_julia_metadata(properties[key])
    catch
        return nothing
    end
end

function _dwave_solver_name(dwave_sampler)
    solver = _maybe_getproperty(dwave_sampler, :solver)
    solver === nothing && return nothing

    for name in (:name, :id)
        value = _as_julia_metadata(_maybe_getproperty(solver, name))
        value === nothing || return value
    end

    return nothing
end

function _dwave_chip_info(dwave_sampler)
    base_sampler = something(_maybe_getproperty(dwave_sampler, :child), dwave_sampler)
    chip_info = Dict{String,Any}()
    properties = _maybe_getproperty(base_sampler, :properties)

    if properties !== nothing
        for key in _DWAVE_CHIP_INFO_KEYS
            value = _metadata_property(properties, key)
            value === nothing && continue

            chip_info[key] = value
        end
    end

    if !haskey(chip_info, "solver_name")
        value = _dwave_solver_name(base_sampler)
        value === nothing || (chip_info["solver_name"] = value)
    end

    return chip_info
end

function _normalise_embedding_index(value)
    value isa Integer && return Int(value)

    if value isa AbstractString
        index = tryparse(Int, value)
        index === nothing || return index
    end

    return nothing
end

function _normalise_embedding_chain(chain)
    chain isa AbstractVector || chain isa Tuple || return nothing

    normal_chain = Int[]
    sizehint!(normal_chain, length(chain))

    for qubit in chain
        index = _normalise_embedding_index(qubit)
        index === nothing && return nothing

        push!(normal_chain, index)
    end

    return normal_chain
end

function _normalise_embedding(embedding)
    embedding isa AbstractDict || return nothing

    normal_embedding = Dict{Int,Vector{Int}}()

    for (variable, chain) in pairs(embedding)
        normal_variable = _normalise_embedding_index(variable)
        normal_variable === nothing && return nothing
        normal_chain = _normalise_embedding_chain(chain)
        normal_chain === nothing && return nothing

        normal_embedding[normal_variable] = normal_chain
    end

    return normal_embedding
end

function _normalise_dwave_embedding!(dwave_info::AbstractDict)
    context = get(dwave_info, "embedding_context", nothing)
    context isa AbstractDict || return dwave_info

    normal_embedding = _normalise_embedding(get(context, "embedding", nothing))
    normal_embedding === nothing || (context["embedding"] = normal_embedding)

    return dwave_info
end

@doc raw"""
    DWave.embedding(sampleset_or_metadata)

Return the minor embedding recorded in D-Wave sample-set metadata, or `nothing`
when no embedding was returned. Embeddings are normalized as
`Dict{Int,Vector{Int}}`.
"""
function embedding(metadata::AbstractDict)
    dwave_info = haskey(metadata, "dwave_info") ? metadata["dwave_info"] : metadata
    dwave_info isa AbstractDict || return nothing

    context = get(dwave_info, "embedding_context", nothing)
    context isa AbstractDict || return nothing

    return _normalise_embedding(get(context, "embedding", nothing))
end

function embedding(sampleset::QUBOTools.SampleSet)
    return embedding(QUBOTools.metadata(sampleset))
end

function QUBODrivers.sample(sampler::Optimizer{T}) where {T}
    # Ising Model
    n, h, J, α, β = QUBOTools.ising(sampler, :dict; sense = :min)

    # Attributes
    num_reads = MOI.get(sampler, DWave.NumberOfReads())
    final_num_reads = MOI.get(sampler, QUBODrivers.FinalNumberOfReads())
    sample_params = Dict{Symbol,Any}(
        :num_reads         => final_num_reads,
        :annealing_time    => MOI.get(sampler, DWave.AnnealingTime()),
        :return_embedding => MOI.get(sampler, DWave.ReturnEmbedding()),
    )
    dwave_sampler = MOI.get(sampler, DWave.Sampler())

    if dwave_sampler === nothing
        dwave_sampler = dwave_system.EmbeddingComposite(
            dwave_system.DWaveSampler(;
                token = get(ENV, "DWAVE_API_TOKEN", nothing)
            )
        )
    end

    # Results
    samples = QUBOTools.Sample{T,Int}[]
    results = @timed dwave_sampler.sample_ising(h, J; sample_params...)
    var_map = pyconvert.(Int, [var for var in results.value.variables])
    dw_info = jl_object(results.value.info)
    _normalise_dwave_embedding!(dw_info)
    chip_info = _dwave_chip_info(dwave_sampler)

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
    metadata = DWave._metadata_base(
        origin = "D-Wave",
        algorithm_name = "D-Wave Quantum Annealing Sampler",
        execution_mode = "qpu",
        number_of_reads = num_reads,
        final_number_of_reads = final_num_reads,
    )
    metadata["time"] = Dict{String,Any}(
        "effective" => DWave._dwave_effective_time(dw_info, results.time),
    )
    DWave._attach_dwave_timing!(metadata, dw_info)
    metadata["dwave_info"] = dw_info

    return QUBOTools.SampleSet{T,Int}(samples, metadata; sense = :min, domain = :spin)
end
