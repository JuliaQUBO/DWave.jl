import DWave
import QUBODrivers
import Random
import Test
import QUBOTools

const MOI = QUBODrivers.MOI
const NealRecord = NamedTuple{(:energy, :reads, :state),Tuple{Float64,Int,Tuple{Vararg{Int}}}}
const NealPyException = DWave.Neal.PythonCall.PyException

function _random_ising_instance(n::Int, seed::Int)
    rng = Random.MersenneTwister(seed)

    h = randn(rng, n)
    J = zeros(Float64, n, n)

    for i in 1:n, j in (i + 1):n
        J[i, j] = randn(rng)
    end

    return h, J
end

function _direct_neal_records(h::Vector{Float64}, J::Matrix{Float64}; kwargs...)
    results = DWave.Neal.dwave_samplers.SimulatedAnnealingSampler().sample_ising(
        DWave.Neal.np.array(h),
        DWave.Neal.np.array(J);
        kwargs...,
    )

    n = length(h)
    var_map = DWave.Neal.PythonCall.pyconvert.(Int, [var for var in results.variables]) .+ 1
    records = NealRecord[]

    for (ϕ, λ, r) in results.record
        ψ = zeros(Int, n)

        for (i, v) in enumerate(ϕ)
            ψ[var_map[i]] = DWave.Neal.PythonCall.pyconvert(Int, v)
        end

        push!(
            records,
            (
                energy = DWave.Neal.PythonCall.pyconvert(Float64, λ),
                reads = DWave.Neal.PythonCall.pyconvert(Int, r),
                state = Tuple(ψ),
            ),
        )
    end

    sort!(records; by = r -> (r.energy, r.reads, r.state))

    return records
end

function _aggregate_records(records::Vector{NealRecord})
    aggregated = Dict{Tuple{Float64,Tuple},Int}()

    for record in records
        key = (record.energy, record.state)
        aggregated[key] = get(aggregated, key, 0) + record.reads
    end

    normalized = NealRecord[]

    for ((energy, state), reads) in aggregated
        push!(normalized, (energy = energy, reads = reads, state = state))
    end

    sort!(normalized; by = r -> (r.energy, r.reads, r.state))

    return normalized
end

function _wrapper_neal_model(h::Vector{Float64}, J::Matrix{Float64}; kwargs...)
    n = length(h)
    model = MOI.instantiate(DWave.Neal.Optimizer; with_bridge_type = Float64)
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

function _wrapper_neal_records(h::Vector{Float64}, J::Matrix{Float64}; kwargs...)
    model, s = _wrapper_neal_model(h, J; kwargs...)
    QUBOTools_MOI = Base.get_extension(QUBOTools, :QUBOTools_MOI)

    @assert !isnothing(QUBOTools_MOI)

    MOI.optimize!(model)

    records = NealRecord[]

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

function _wrapper_neal_error(h::Vector{Float64}, J::Matrix{Float64}; kwargs...)
    model, _ = _wrapper_neal_model(h, J; kwargs...)

    try
        MOI.optimize!(model)
        return nothing
    catch err
        return err
    end
end

function _reset_neal_python_modules!()
    DWave.Neal.PythonCall.pyexec(
        """
for name in [name for name in list(sys.modules) if name == "dwave.samplers" or name.startswith("dwave.samplers.")]:
    del sys.modules[name]

if hasattr(dwave, "samplers"):
    del dwave.samplers
""",
        @__MODULE__,
        (
            dwave = DWave.Neal.PythonCall.pyimport("dwave"),
            sys = DWave.Neal.PythonCall.pyimport("sys"),
        ),
    )

    DWave.Neal.__init__()

    return nothing
end

function _sys_modules_contains(name::String)
    sys = DWave.Neal.PythonCall.pyimport("sys")
    return DWave.Neal.PythonCall.pyconvert(Bool, sys.modules.__contains__(name))
end

Test.@testset "Neal initialization loads the simulated annealing sampler" begin
    _reset_neal_python_modules!()

    Test.@test DWave.Neal.PythonCall.pyconvert(String, DWave.Neal.dwave_samplers.__name__) ==
        "dwave.samplers.sa.sampler"
    Test.@test _sys_modules_contains("dwave.samplers")
    Test.@test _sys_modules_contains("dwave.samplers.sa")
    Test.@test _sys_modules_contains("dwave.samplers.sa.sampler")
    Test.@test _sys_modules_contains("dwave.samplers.sa.simulated_annealing")
    Test.@test DWave.Neal.dwave_samplers.SimulatedAnnealingSampler !== nothing
    Test.@test DWave.Neal.dwave_samplers_import_mode[] in (:narrow, :fallback)

    if DWave.Neal.dwave_samplers_import_mode[] == :narrow
        Test.@test !_sys_modules_contains("dwave.samplers.random")
    else
        Test.@test DWave.Neal.dwave_samplers_import_mode[] == :fallback
    end
end

Test.@testset "Neal parity with direct dwave.samplers" begin
    h, J = _random_ising_instance(100, 42)

    for schedule in ("geometric", "linear")
        Test.@testset "schedule=$schedule" begin
            kwargs = (
                num_reads = 128,
                num_sweeps = 85,
                beta_schedule_type = schedule,
                seed = 123_456,
            )

            direct_records = _aggregate_records(_direct_neal_records(h, J; kwargs...))
            wrapper_records = _aggregate_records(_wrapper_neal_records(h, J; kwargs...))

            Test.@test wrapper_records == direct_records
        end
    end
end

Test.@testset "Neal wrapper seed determinism" begin
    h, J = _random_ising_instance(40, 314)
    kwargs = (
        num_reads = 64,
        num_sweeps = 50,
        beta_schedule_type = "geometric",
        seed = 202_603_27,
    )

    records_a = _aggregate_records(_wrapper_neal_records(h, J; kwargs...))
    records_b = _aggregate_records(_wrapper_neal_records(h, J; kwargs...))

    Test.@test records_a == records_b
end

Test.@testset "Neal wrapper schedule validation" begin
    h = [0.0, -1.0]
    J = zeros(Float64, 2, 2)
    J[1, 2] = -1.0

    for (kwargs, message) in (
        ((; num_reads = 1, beta_schedule_type = "asd"), "Beta schedule type asd not implemented"),
        ((; num_reads = 1, beta_schedule_type = "custom"), "'beta_schedule' must be provided"),
        (
            (; num_reads = 1, beta_schedule_type = "linear", beta_schedule = [0.1, 1.0]),
            "'beta_schedule' must be set to None",
        ),
    )
        err = _wrapper_neal_error(h, J; kwargs...)

        Test.@test err isa NealPyException
        Test.@test occursin(message, err === nothing ? "" : sprint(showerror, err))
    end
end

Test.@testset "Neal parity with sparse support" begin
    h = zeros(Float64, 8)
    h[2] = -0.75
    h[7] = 0.25

    J = zeros(Float64, 8, 8)
    J[1, 5] = -1.0
    J[3, 4] = 0.5

    kwargs = (
        num_reads = 64,
        num_sweeps = 40,
        beta_schedule_type = "geometric",
        seed = 9_191,
    )

    direct_records = _aggregate_records(_direct_neal_records(h, J; kwargs...))
    wrapper_records = _aggregate_records(_wrapper_neal_records(h, J; kwargs...))

    Test.@test wrapper_records == direct_records
    Test.@test all(all(abs.(record.state) .== 1) for record in wrapper_records)
end
