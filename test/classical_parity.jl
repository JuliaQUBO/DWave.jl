import DWave
import QUBODrivers
import QUBOTools
import Random
import Test

const MOI = QUBODrivers.MOI
const ClassicalRecord = NamedTuple{(:energy, :reads, :state),Tuple{Float64,Int,Tuple{Vararg{Int}}}}

const CLASSICAL_SPECS = (
    (
        name = "Greedy",
        sampler_module = DWave.Greedy,
        optimizer = DWave.Greedy.Optimizer,
        constructor = :SteepestDescentSampler,
        module_name = "dwave.samplers.greedy.sampler",
        present_modules = (
            "dwave.samplers",
            "dwave.samplers.greedy",
            "dwave.samplers.greedy.sampler",
            "dwave.samplers.greedy.descent",
        ),
        absent_modules = (
            "dwave.samplers.random",
            "dwave.samplers.sa",
            "dwave.samplers.tabu",
        ),
    ),
    (
        name = "Random",
        sampler_module = DWave.Random,
        optimizer = DWave.Random.Optimizer,
        constructor = :RandomSampler,
        module_name = "dwave.samplers.random.sampler",
        present_modules = (
            "dwave.samplers",
            "dwave.samplers.random",
            "dwave.samplers.random.sampler",
            "dwave.samplers.random.cyrandom",
        ),
        absent_modules = (
            "dwave.samplers.greedy",
            "dwave.samplers.sa",
            "dwave.samplers.tabu",
        ),
    ),
    (
        name = "Tabu",
        sampler_module = DWave.Tabu,
        optimizer = DWave.Tabu.Optimizer,
        constructor = :TabuSampler,
        module_name = "dwave.samplers.tabu",
        present_modules = (
            "dwave.samplers",
            "dwave.samplers.tabu",
            "dwave.samplers.tabu.sampler",
            "dwave.samplers.tabu.tabu_search",
        ),
        absent_modules = (
            "dwave.samplers.greedy",
            "dwave.samplers.random",
            "dwave.samplers.sa",
        ),
    ),
)

function _classical_random_ising_instance(n::Int, seed::Int)
    rng = Random.MersenneTwister(seed)

    h = randn(rng, n)
    J = zeros(Float64, n, n)

    for i in 1:n, j in (i + 1):n
        J[i, j] = randn(rng)
    end

    return h, J
end

function _classical_aggregate_records(records::Vector{ClassicalRecord})
    aggregated = Dict{Tuple{Float64,Tuple},Int}()

    for record in records
        key = (record.energy, record.state)
        aggregated[key] = get(aggregated, key, 0) + record.reads
    end

    normalized = ClassicalRecord[]

    for ((energy, state), reads) in aggregated
        push!(normalized, (energy = energy, reads = reads, state = state))
    end

    sort!(normalized; by = r -> (r.energy, r.reads, r.state))

    return normalized
end

function _direct_records(spec, h::Vector{Float64}, J::Matrix{Float64}; kwargs...)
    results = getproperty(spec.sampler_module.dwave_samplers, spec.constructor)().sample_ising(
        spec.sampler_module.PythonCall.Py(h),
        spec.sampler_module.PythonCall.Py(J);
        kwargs...,
    )

    n = length(h)
    var_map = spec.sampler_module.PythonCall.pyconvert.(Int, [var for var in results.variables]) .+ 1
    records = ClassicalRecord[]

    for row in results.record
        ψ = zeros(Int, n)

        for (i, v) in enumerate(row["sample"])
            ψ[var_map[i]] = spec.sampler_module.PythonCall.pyconvert(Int, v)
        end

        push!(
            records,
            (
                energy = spec.sampler_module.PythonCall.pyconvert(Float64, row["energy"]),
                reads = spec.sampler_module.PythonCall.pyconvert(Int, row["num_occurrences"]),
                state = Tuple(ψ),
            ),
        )
    end

    sort!(records; by = r -> (r.energy, r.reads, r.state))

    return records
end

function _wrapper_model(optimizer, h::Vector{Float64}, J::Matrix{Float64}; kwargs...)
    n = length(h)
    model = MOI.instantiate(optimizer; with_bridge_type = Float64)
    s, _ = MOI.add_constrained_variables(model, fill(QUBODrivers.Spin(), n))

    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(
        model,
        MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(),
        MOI.ScalarQuadraticFunction{Float64}(
            MOI.ScalarQuadraticTerm{Float64}[
                MOI.ScalarQuadraticTerm{Float64}(J[i, j], s[i], s[j]) for i in 1:n for j in (i + 1):n
            ],
            MOI.ScalarAffineTerm{Float64}[MOI.ScalarAffineTerm{Float64}(h[i], s[i]) for i in 1:n],
            0.0,
        ),
    )

    for (name, value) in pairs(kwargs)
        MOI.set(model, MOI.RawOptimizerAttribute(String(name)), value)
    end

    return model, s
end

function _wrapper_records(optimizer, h::Vector{Float64}, J::Matrix{Float64}; kwargs...)
    model, s = _wrapper_model(optimizer, h, J; kwargs...)
    QUBOTools_MOI = Base.get_extension(QUBOTools, :QUBOTools_MOI)

    @assert !isnothing(QUBOTools_MOI)

    MOI.optimize!(model)

    records = ClassicalRecord[]

    for i in 1:MOI.get(model, MOI.ResultCount())
        push!(
            records,
            (
                energy = MOI.get(model, MOI.ObjectiveValue(i)),
                reads = MOI.get(model, QUBOTools_MOI.NumberOfReads(i)),
                state = Tuple(Int(round(MOI.get(model, MOI.VariablePrimal(i), v))) for v in s),
            ),
        )
    end

    sort!(records; by = r -> (r.energy, r.reads, r.state))

    return records
end

function _classical_sys_modules_contains(name::String)
    sys = DWave.Neal.PythonCall.pyimport("sys")
    return DWave.Neal.PythonCall.pyconvert(Bool, sys.modules.__contains__(name))
end

for spec in CLASSICAL_SPECS
    Test.@testset "$(spec.name) initialization isolates the sampler import tree" begin
        DWave._clear_dwave_samplers_import_state!()
        spec.sampler_module.__init__()

        Test.@test spec.sampler_module.PythonCall.pyconvert(
            String,
            spec.sampler_module.dwave_samplers.__name__,
        ) ==
            spec.module_name

        for module_name in spec.present_modules
            Test.@test _classical_sys_modules_contains(module_name)
        end

        if Sys.iswindows()
            Test.@test spec.sampler_module.dwave_samplers_import_mode[] in (:narrow, :fallback)
        else
            Test.@test spec.sampler_module.dwave_samplers_import_mode[] == :narrow

            for module_name in spec.absent_modules
                Test.@test !_classical_sys_modules_contains(module_name)
            end
        end
    end
end

Test.@testset "Greedy parity with direct dwave.samplers" begin
    h, J = _classical_random_ising_instance(64, 17)
    kwargs = (
        num_reads = 96,
        seed = 202_603_28,
        large_sparse_opt = true,
    )

    direct_records = _classical_aggregate_records(_direct_records(CLASSICAL_SPECS[1], h, J; kwargs...))
    wrapper_records = _classical_aggregate_records(
        _wrapper_records(DWave.Greedy.Optimizer, h, J; kwargs...),
    )

    Test.@test wrapper_records == direct_records
end

Test.@testset "Greedy wrapper forwards initial states" begin
    h = [0.0, 0.0]
    J = zeros(Float64, 2, 2)
    J[1, 2] = 1.0

    kwargs = (
        initial_states = [-1 -1],
        initial_states_generator = "tile",
        num_reads = 6,
        seed = 91,
    )

    direct_records = _classical_aggregate_records(_direct_records(CLASSICAL_SPECS[1], h, J; kwargs...))
    wrapper_records = _classical_aggregate_records(
        _wrapper_records(DWave.Greedy.Optimizer, h, J; kwargs...),
    )

    Test.@test wrapper_records == direct_records
end

Test.@testset "Random parity with direct dwave.samplers" begin
    h, J = _classical_random_ising_instance(20, 29)
    kwargs = (
        num_reads = 80,
        seed = 202_603_29,
        max_num_samples = 16,
    )

    direct_records = _classical_aggregate_records(_direct_records(CLASSICAL_SPECS[2], h, J; kwargs...))
    wrapper_records = _classical_aggregate_records(
        _wrapper_records(DWave.Random.Optimizer, h, J; kwargs...),
    )

    Test.@test wrapper_records == direct_records
    Test.@test sum(record.reads for record in wrapper_records) == kwargs.num_reads
end

Test.@testset "Random wrapper seed determinism" begin
    h, J = _classical_random_ising_instance(24, 37)
    kwargs = (
        num_reads = 48,
        seed = 202_603_30,
    )

    records_a = _classical_aggregate_records(_wrapper_records(DWave.Random.Optimizer, h, J; kwargs...))
    records_b = _classical_aggregate_records(_wrapper_records(DWave.Random.Optimizer, h, J; kwargs...))

    Test.@test records_a == records_b
end

Test.@testset "Tabu parity with direct dwave.samplers" begin
    h, J = _classical_random_ising_instance(18, 43)
    kwargs = (
        num_reads = 24,
        seed = 202_603_31,
        timeout = nothing,
        num_restarts = 2,
    )

    direct_records = _classical_aggregate_records(_direct_records(CLASSICAL_SPECS[3], h, J; kwargs...))
    wrapper_records = _classical_aggregate_records(
        _wrapper_records(DWave.Tabu.Optimizer, h, J; kwargs...),
    )

    Test.@test wrapper_records == direct_records
end

Test.@testset "Tabu wrapper forwards initial states" begin
    h = [0.0, 0.0, 0.0]
    J = zeros(Float64, 3, 3)
    J[1, 2] = -1.0
    J[1, 3] = 1.0
    J[2, 3] = 1.0

    kwargs = (
        initial_states = [1 1 1; -1 -1 -1],
        initial_states_generator = "tile",
        num_reads = 5,
        seed = 123,
        timeout = nothing,
        num_restarts = 0,
    )

    direct_records = _classical_aggregate_records(_direct_records(CLASSICAL_SPECS[3], h, J; kwargs...))
    wrapper_records = _classical_aggregate_records(
        _wrapper_records(DWave.Tabu.Optimizer, h, J; kwargs...),
    )

    Test.@test wrapper_records == direct_records
end
