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

const _DWAVE_CHIP_INFO_KEYS = ("chip_id", "topology", "solver_name", "category")

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
            dwave_system.DWaveSampler(;
                token = get(ENV, "DWAVE_API_TOKEN", nothing)
            )
        )

        sample_params[:return_embedding] = MOI.get(sampler, DWave.ReturnEmbedding())
    end

    # Results
    samples = QUBOTools.Sample{T,Int}[]
    results = @timed dwave_sampler.sample_ising(h, J; sample_params...)
    var_map = pyconvert.(Int, [var for var in results.value.variables])
    dw_info = jl_object(results.value.info)
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
    metadata = Dict{String,Any}(
        "origin" => "D-Wave",
        "time"   => Dict{String,Any}( #
            "effective" => results.time,
        ),
        "dwave_info" => dw_info,
    )

    return QUBOTools.SampleSet{T,Int}(samples, metadata; sense = :min, domain = :spin)
end
