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

function _wrapper_neal_sampleset(h::Vector{Float64}, J::Matrix{Float64}; kwargs...)
    model, _ = _wrapper_neal_model(h, J; kwargs...)
    MOI.optimize!(model)
    return QUBOTools.solution(MOI.get(model, MOI.RawSolver()))
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

function _install_recording_neal_sampler!()
    original = DWave.Neal.dwave_samplers.SimulatedAnnealingSampler

    DWave.Neal.PythonCall.pyexec(
        """
def make_recording_sampler(dimod, np):
    class RecordingSimulatedAnnealingSampler:
        last = None

        def sample(self, bqm, **params):
            variable_order = list(bqm.variables)
            vectors = bqm.to_numpy_vectors(variable_order=variable_order)
            init_states = params.get("initial_states")
            if init_states is None:
                recorded_states = None
                recorded_state_vars = None
            else:
                state_rows, state_vars = init_states
                recorded_states = np.asarray(state_rows, dtype=int).tolist()
                recorded_state_vars = [int(v) for v in state_vars]
            RecordingSimulatedAnnealingSampler.last = {
                "num_variables": bqm.num_variables,
                "num_interactions": bqm.num_interactions,
                "variables": variable_order,
                "rows": vectors.quadratic.row_indices.tolist(),
                "cols": vectors.quadratic.col_indices.tolist(),
                "weights": vectors.quadratic.biases.tolist(),
                "initial_states": recorded_states,
                "initial_state_variables": recorded_state_vars,
                "initial_states_generator": params.get("initial_states_generator"),
            }
            samples = np.ones((1, bqm.num_variables), dtype=np.int8)
            return dimod.SampleSet.from_samples(
                (samples, variable_order),
                energy=[0.0],
                vartype=dimod.SPIN,
                info={
                    "beta_range": [0.1, 1.0],
                    "beta_schedule_type": params.get("beta_schedule_type", "geometric"),
                    "timing": {},
                },
            )

        def sample_ising(self, *args, **params):
            raise AssertionError("Neal wrapper should call sample with a sparse BQM")

    return RecordingSimulatedAnnealingSampler

target_module.SimulatedAnnealingSampler = make_recording_sampler(dimod, np)
""",
        @__MODULE__,
        (
            target_module = DWave.Neal.dwave_samplers,
            dimod = DWave.dwave_dimod,
            np = DWave.Neal.np,
        ),
    )

    return original
end

function _restore_neal_sampler!(original)
    DWave.Neal.PythonCall.pyexec(
        "target_module.SimulatedAnnealingSampler = original",
        @__MODULE__,
        (target_module = DWave.Neal.dwave_samplers, original = original),
    )

    return nothing
end

function _recording_neal_sampler_state()
    record = DWave.Neal.dwave_samplers.SimulatedAnnealingSampler.last
    py_states = record["initial_states"]
    py_state_vars = record["initial_state_variables"]
    py_gen = record["initial_states_generator"]

    return (
        num_variables = DWave.Neal.PythonCall.pyconvert(Int, record["num_variables"]),
        num_interactions = DWave.Neal.PythonCall.pyconvert(Int, record["num_interactions"]),
        variables = DWave.Neal.PythonCall.pyconvert(Vector{Int}, record["variables"]),
        rows = DWave.Neal.PythonCall.pyconvert(Vector{Int}, record["rows"]),
        cols = DWave.Neal.PythonCall.pyconvert(Vector{Int}, record["cols"]),
        weights = DWave.Neal.PythonCall.pyconvert(Vector{Float64}, record["weights"]),
        initial_states = DWave.Neal.PythonCall.pyis(py_states, DWave.Neal.PythonCall.pybuiltins.None) ?
                         nothing :
                         DWave.Neal.PythonCall.pyconvert(Vector{Vector{Int}}, py_states),
        initial_state_variables = DWave.Neal.PythonCall.pyis(
            py_state_vars,
            DWave.Neal.PythonCall.pybuiltins.None,
        ) ? nothing : DWave.Neal.PythonCall.pyconvert(Vector{Int}, py_state_vars),
        initial_states_generator = DWave.Neal.PythonCall.pyis(
            py_gen,
            DWave.Neal.PythonCall.pybuiltins.None,
        ) ? nothing : DWave.Neal.PythonCall.pyconvert(String, py_gen),
    )
end

function _reset_neal_python_modules!()
    DWave.Neal._clear_sa_import_state!()
    # A failure here means the Neal import path could not be rebuilt cleanly.
    DWave.Neal.__init__()

    return nothing
end

function _inject_broken_sa_sampler!()
    DWave.Neal._clear_sa_import_state!()

    DWave.Neal.PythonCall.pyexec(
        """
samplers_name = "dwave.samplers"
sa_name = "dwave.samplers.sa"
sampler_name = "dwave.samplers.sa.sampler"

samplers_pkg = types.ModuleType(samplers_name)
samplers_pkg.__path__ = []
samplers_pkg.__package__ = samplers_name
sys.modules[samplers_name] = samplers_pkg
dwave.samplers = samplers_pkg

sa_pkg = types.ModuleType(sa_name)
sa_pkg.__path__ = []
sa_pkg.__package__ = sa_name
sys.modules[sa_name] = sa_pkg
samplers_pkg.sa = sa_pkg

sampler = types.ModuleType(sampler_name)
sys.modules[sampler_name] = sampler
sa_pkg.sampler = sampler
""",
        @__MODULE__,
        (
            dwave = DWave.Neal.PythonCall.pyimport("dwave"),
            sys = DWave.Neal.PythonCall.pyimport("sys"),
            types = DWave.Neal.PythonCall.pyimport("types"),
        ),
    )

    return nothing
end

function _sys_modules_contains(name::String)
    sys = DWave.Neal.PythonCall.pyimport("sys")
    return DWave.Neal.PythonCall.pyconvert(Bool, sys.modules.__contains__(name))
end

function _assert_neal_sampler_works()
    h = [0.0, -1.0]
    J = zeros(Float64, 2, 2)
    J[1, 2] = -1.0

    records = _direct_neal_records(h, J; num_reads = 1, num_sweeps = 10, seed = 314_159)

    Test.@test length(records) == 1
    Test.@test isfinite(only(records).energy)
    Test.@test all(abs.(collect(only(records).state)) .== 1)

    return nothing
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

    if Sys.iswindows()
        Test.@test DWave.Neal.dwave_samplers_import_mode[] in (:narrow, :fallback)
    else
        Test.@test DWave.Neal.dwave_samplers_import_mode[] == :narrow
        Test.@test !_sys_modules_contains("dwave.samplers.random")
    end

    _assert_neal_sampler_works()
end

Test.@testset "Neal initialization repairs broken sampler cache state" begin
    _inject_broken_sa_sampler!()
    DWave.Neal.__init__()

    Test.@test DWave.Neal.PythonCall.pyconvert(String, DWave.Neal.dwave_samplers.__name__) ==
        "dwave.samplers.sa.sampler"
    Test.@test DWave.Neal.dwave_samplers.SimulatedAnnealingSampler !== nothing

    if Sys.iswindows()
        Test.@test DWave.Neal.dwave_samplers_import_mode[] in (:narrow, :fallback)
    else
        Test.@test DWave.Neal.dwave_samplers_import_mode[] == :narrow
        Test.@test !_sys_modules_contains("dwave.samplers.random")
    end

    Test.@test _sys_modules_contains("dwave.samplers.sa.simulated_annealing")
    _assert_neal_sampler_works()
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

Test.@testset "Neal wrapper builds sparse BQM input" begin
    h = zeros(Float64, 8)
    h[2] = -0.75
    h[7] = 0.25

    J = zeros(Float64, 8, 8)
    J[1, 5] = -1.0
    J[3, 4] = 0.5

    original = _install_recording_neal_sampler!()

    try
        _wrapper_neal_records(h, J; num_reads = 1, num_sweeps = 10, seed = 11)

        record = _recording_neal_sampler_state()

        Test.@test record.num_variables == 8
        Test.@test record.num_interactions == 2
        Test.@test record.variables == collect(0:7)
        Test.@test record.rows == [3, 4]
        Test.@test record.cols == [2, 0]
        Test.@test record.weights == [0.5, -1.0]
    finally
        _restore_neal_sampler!(original)
    end
end

Test.@testset "Neal metadata includes solver info" begin
    h = [0.0, -1.0]
    J = zeros(Float64, 2, 2)
    J[1, 2] = -1.0

    sampleset = _wrapper_neal_sampleset(h, J; num_reads = 4, num_sweeps = 10, seed = 7)
    metadata = QUBOTools.metadata(sampleset)

    Test.@test haskey(metadata, "origin")
    Test.@test haskey(metadata, "algorithm")
    Test.@test haskey(metadata, "backend")
    Test.@test haskey(metadata, "reads")
    Test.@test haskey(metadata, "seeds")
    Test.@test haskey(metadata, "time")
    Test.@test haskey(metadata, "dwave_info")
    Test.@test isempty(QUBODrivers.validate_metadata(metadata))

    Test.@test metadata["algorithm"]["name"] == "D-Wave Neal Simulated Annealing Sampler"
    Test.@test metadata["backend"]["name"] == "dwave-ocean-sdk"
    Test.@test metadata["backend"]["version"] == DWave.OCEAN_SDK_VERSION
    Test.@test metadata["reads"]["number_of_reads"] == 4
    Test.@test metadata["reads"]["final_number_of_reads"] == 4
    Test.@test metadata["seeds"]["sampler"] == 7
    Test.@test metadata["time"]["effective"] > 0
    info = metadata["dwave_info"]
    Test.@test haskey(info, "beta_range")
    Test.@test haskey(info, "beta_schedule_type")
    Test.@test haskey(info, "timing")
    Test.@test length(info["beta_range"]) == 2
    Test.@test info["timing"] isa Dict
end

function _wrapper_neal_qubo_model(
    linear::Vector{Float64},
    edges::Vector{Tuple{Int,Int}},
    quadratic::Vector{Float64},
    n::Int;
    domain::Symbol = :bool,
    sense::Symbol = :min,
    offset::Float64 = 0.0,
    kwargs...,
)
    model = MOI.instantiate(DWave.Neal.Optimizer; with_bridge_type = Float64)
    set = domain === :bool ? MOI.ZeroOne() : QUBODrivers.Spin()
    x, _ = MOI.add_constrained_variables(model, fill(set, n))

    MOI.set(model, MOI.ObjectiveSense(), sense === :min ? MOI.MIN_SENSE : MOI.MAX_SENSE)
    MOI.set(
        model,
        MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(),
        MOI.ScalarQuadraticFunction{Float64}(
            MOI.ScalarQuadraticTerm{Float64}[
                MOI.ScalarQuadraticTerm{Float64}(quadratic[k], x[i], x[j])
                for (k, (i, j)) in enumerate(edges)
            ],
            MOI.ScalarAffineTerm{Float64}[
                MOI.ScalarAffineTerm{Float64}(linear[i], x[i]) for i in eachindex(linear)
            ],
            offset,
        ),
    )

    for (name, value) in pairs(kwargs)
        MOI.set(model, MOI.RawOptimizerAttribute(String(name)), value)
    end

    return model, x
end

Test.@testset "Neal wrapper forwards initial_states across domains, shapes, and isolated variables" begin
    # Variable 4 is isolated (zero linear and quadratic bias).
    linear = [0.5, -0.75, 0.25]
    edges = [(1, 2), (2, 3)]
    quadratic = [1.0, -0.5]
    n = 4
    offset = 1.25
    energy_fn = x -> sum(linear[i] * x[i] for i in eachindex(linear)) +
        sum(quadratic[k] * x[i] * x[j] for (k, (i, j)) in enumerate(edges)) + offset

    # Single-vector initial_states in :bool (including all-1 state that would
    # otherwise be ambiguous between SPIN and BINARY in dimod's infer_vartype)
    for init_vec in ([1, 1, 1, 1], [0, 1, 0, 1], [1, 0, 1, 0])
        model, x = _wrapper_neal_qubo_model(
            linear,
            edges,
            quadratic,
            n;
            domain = :bool,
            offset = offset,
            num_reads = 1,
            num_sweeps = 0,
            initial_states = init_vec,
            initial_states_generator = "none",
        )
        MOI.optimize!(model)
        Test.@test MOI.get(model, MOI.TerminationStatus()) == MOI.LOCALLY_SOLVED
        Test.@test MOI.get(model, MOI.ResultCount()) == 1
        state = [Int(round(MOI.get(model, MOI.VariablePrimal(1), v))) for v in x]
        Test.@test state == init_vec
        Test.@test MOI.get(model, MOI.ObjectiveValue(1)) ≈ energy_fn(init_vec) atol = 1e-14
    end

    # Single-vector initial_states in :spin (preserving isolated variable 4)
    for init_vec in ([1, 1, 1, 1], [-1, 1, -1, 1], [1, -1, 1, -1])
        model, s = _wrapper_neal_qubo_model(
            linear,
            edges,
            quadratic,
            n;
            domain = :spin,
            offset = offset,
            num_reads = 1,
            num_sweeps = 0,
            initial_states = init_vec,
            initial_states_generator = "none",
        )
        MOI.optimize!(model)
        Test.@test MOI.get(model, MOI.TerminationStatus()) == MOI.LOCALLY_SOLVED
        Test.@test MOI.get(model, MOI.ResultCount()) == 1
        state = [Int(round(MOI.get(model, MOI.VariablePrimal(1), v))) for v in s]
        Test.@test state == init_vec
        Test.@test MOI.get(model, MOI.ObjectiveValue(1)) ≈ energy_fn(init_vec) atol = 1e-14
    end

    # Multi-row matrix initial_states in :bool and :spin
    bool_batch = [
        0 1 0 1
        1 0 1 0
        1 1 1 1
    ]
    model_bool, x_bool = _wrapper_neal_qubo_model(
        linear,
        edges,
        quadratic,
        n;
        domain = :bool,
        offset = offset,
        num_reads = 3,
        num_sweeps = 0,
        initial_states = bool_batch,
        initial_states_generator = "none",
    )
    MOI.optimize!(model_bool)
    Test.@test MOI.get(model_bool, MOI.ResultCount()) == 3
    returned_bool = Set(
        Tuple(Int(round(MOI.get(model_bool, MOI.VariablePrimal(i), v))) for v in x_bool)
        for i in 1:MOI.get(model_bool, MOI.ResultCount())
    )
    expected_bool = Set(Tuple(bool_batch[row, :]) for row in 1:size(bool_batch, 1))
    Test.@test returned_bool == expected_bool

    # Native-call capture proving exact payload forwarding for single-state vector
    # and row-oriented batch across :bool and :spin (including isolated variable 4)
    original = _install_recording_neal_sampler!()
    try
        m_default, _ = _wrapper_neal_qubo_model(linear, edges, quadratic, n; domain = :bool, num_reads = 1)
        MOI.optimize!(m_default)
        rec_default = _recording_neal_sampler_state()
        Test.@test rec_default.initial_states === nothing
        Test.@test rec_default.initial_state_variables === nothing
        Test.@test rec_default.initial_states_generator == "random"

        m_bvec, _ = _wrapper_neal_qubo_model(
            linear,
            edges,
            quadratic,
            n;
            domain = :bool,
            num_reads = 1,
            initial_states = [0, 1, 0, 1],
            initial_states_generator = "none",
        )
        MOI.optimize!(m_bvec)
        rec_bvec = _recording_neal_sampler_state()
        Test.@test rec_bvec.initial_states == [[-1, 1, -1, 1]]
        Test.@test rec_bvec.initial_state_variables == [0, 1, 2, 3]
        Test.@test rec_bvec.initial_states_generator == "none"

        m_svec, _ = _wrapper_neal_qubo_model(
            linear,
            edges,
            quadratic,
            n;
            domain = :spin,
            num_reads = 1,
            initial_states = [1, -1, 1, -1],
            initial_states_generator = "tile",
        )
        MOI.optimize!(m_svec)
        rec_svec = _recording_neal_sampler_state()
        Test.@test rec_svec.initial_states == [[1, -1, 1, -1]]
        Test.@test rec_svec.initial_state_variables == [0, 1, 2, 3]
        Test.@test rec_svec.initial_states_generator == "tile"

        m_bmat, _ = _wrapper_neal_qubo_model(
            linear,
            edges,
            quadratic,
            n;
            domain = :bool,
            num_reads = 3,
            initial_states = bool_batch,
            initial_states_generator = "tile",
        )
        MOI.optimize!(m_bmat)
        rec_bmat = _recording_neal_sampler_state()
        Test.@test rec_bmat.initial_states == [[-1, 1, -1, 1], [1, -1, 1, -1], [1, 1, 1, 1]]
        Test.@test rec_bmat.initial_state_variables == [0, 1, 2, 3]
        Test.@test rec_bmat.initial_states_generator == "tile"

        m_smat, _ = _wrapper_neal_qubo_model(
            linear,
            edges,
            quadratic,
            n;
            domain = :spin,
            num_reads = 2,
            initial_states = [1 -1 1 -1; -1 1 -1 1],
            initial_states_generator = "none",
        )
        MOI.optimize!(m_smat)
        rec_smat = _recording_neal_sampler_state()
        Test.@test rec_smat.initial_states == [[1, -1, 1, -1], [-1, 1, -1, 1]]
        Test.@test rec_smat.initial_state_variables == [0, 1, 2, 3]
        Test.@test rec_smat.initial_states_generator == "none"
    finally
        _restore_neal_sampler!(original)
    end

    # Direct parity with dwave.samplers.SimulatedAnnealingSampler for nonzero sweeps
    h = [0.25, -0.5, 0.75, 0.0]
    J = zeros(Float64, 4, 4)
    J[1, 2] = -1.0
    J[2, 3] = 0.5
    spin_batch = [
         1 -1  1 -1
        -1  1 -1  1
         1  1 -1  1
    ]
    kwargs = (
        initial_states = spin_batch,
        initial_states_generator = "tile",
        num_reads = 8,
        num_sweeps = 25,
        beta_schedule_type = "geometric",
        seed = 202_604_15,
    )
    direct_records = _aggregate_records(
        _direct_neal_records(
            h,
            J;
            initial_states = DWave.Neal.PythonCall.Py((spin_batch, collect(0:3))),
            initial_states_generator = kwargs.initial_states_generator,
            num_reads = kwargs.num_reads,
            num_sweeps = kwargs.num_sweeps,
            beta_schedule_type = kwargs.beta_schedule_type,
            seed = kwargs.seed,
        ),
    )
    wrapper_records = _aggregate_records(_wrapper_neal_records(h, J; kwargs...))
    Test.@test wrapper_records == direct_records
end

Test.@testset "Neal initial_states generator, seed, and FinalNumberOfReads semantics" begin
    h = [0.5, -0.25, 0.75, 0.0]
    J = zeros(Float64, 4, 4)
    J[1, 2] = -0.5
    J[2, 3] = 0.25
    batch_two = [
         1 -1  1 -1
        -1  1 -1  1
    ]

    # 1. M < final_num_reads with generator = "none" raises a Python ValueError
    model_none, _ = _wrapper_neal_model(
        h,
        J;
        num_reads = 6,
        num_sweeps = 0,
        initial_states = batch_two,
        initial_states_generator = "none",
    )
    MOI.set(model_none, QUBODrivers.FinalNumberOfReads(), 5)
    err = try
        MOI.optimize!(model_none)
        nothing
    catch caught
        caught
    end
    Test.@test err isa NealPyException
    Test.@test occursin("insufficient number of initial states given", sprint(showerror, err))

    # 2. M < final_num_reads with generator = "tile" repeats states cyclically to final_num_reads
    model_tile, s_tile = _wrapper_neal_model(
        h,
        J;
        num_reads = 9,
        num_sweeps = 0,
        initial_states = batch_two,
        initial_states_generator = "tile",
    )
    MOI.set(model_tile, QUBODrivers.FinalNumberOfReads(), 5)
    MOI.optimize!(model_tile)
    QUBOTools_MOI = Base.get_extension(QUBOTools, :QUBOTools_MOI)
    counts_tile = Dict(
        Tuple(Int(round(MOI.get(model_tile, MOI.VariablePrimal(i), v))) for v in s_tile) =>
            MOI.get(model_tile, QUBOTools_MOI.NumberOfReads(i))
        for i in 1:MOI.get(model_tile, MOI.ResultCount())
    )
    Test.@test counts_tile[Tuple(batch_two[1, :])] == 3
    Test.@test counts_tile[Tuple(batch_two[2, :])] == 2
    metadata_tile = QUBOTools.metadata(QUBOTools.solution(MOI.get(model_tile, MOI.RawSolver())))
    Test.@test metadata_tile["reads"]["number_of_reads"] == 9
    Test.@test metadata_tile["reads"]["final_number_of_reads"] == 5
    Test.@test metadata_tile["reads"]["completed_number_of_reads"] == 5

    # 3. M < final_num_reads with generator = "random" pads remaining reads deterministically with seed
    model_rand_a, s_rand_a = _wrapper_neal_model(
        h,
        J;
        num_reads = 9,
        num_sweeps = 0,
        initial_states = batch_two,
        initial_states_generator = "random",
        seed = 867_5309,
    )
    MOI.set(model_rand_a, QUBODrivers.FinalNumberOfReads(), 5)
    MOI.optimize!(model_rand_a)

    model_rand_b, s_rand_b = _wrapper_neal_model(
        h,
        J;
        num_reads = 9,
        num_sweeps = 0,
        initial_states = batch_two,
        initial_states_generator = "random",
        seed = 867_5309,
    )
    MOI.set(model_rand_b, QUBODrivers.FinalNumberOfReads(), 5)
    MOI.optimize!(model_rand_b)

    counts_rand_a = Dict(
        Tuple(Int(round(MOI.get(model_rand_a, MOI.VariablePrimal(i), v))) for v in s_rand_a) =>
            MOI.get(model_rand_a, QUBOTools_MOI.NumberOfReads(i))
        for i in 1:MOI.get(model_rand_a, MOI.ResultCount())
    )
    counts_rand_b = Dict(
        Tuple(Int(round(MOI.get(model_rand_b, MOI.VariablePrimal(i), v))) for v in s_rand_b) =>
            MOI.get(model_rand_b, QUBOTools_MOI.NumberOfReads(i))
        for i in 1:MOI.get(model_rand_b, MOI.ResultCount())
    )
    Test.@test counts_rand_a == counts_rand_b
    Test.@test sum(values(counts_rand_a)) == 5
    Test.@test get(counts_rand_a, Tuple(batch_two[1, :]), 0) >= 1
    Test.@test get(counts_rand_a, Tuple(batch_two[2, :]), 0) >= 1

    # 4. M > final_num_reads truncates to the first final_num_reads states across all generators
    batch_three = [
         1 -1  1 -1
        -1  1 -1  1
         1  1 -1  1
    ]
    for gen in ("none", "tile", "random")
        model_trunc, s_trunc = _wrapper_neal_model(
            h,
            J;
            num_reads = 6,
            num_sweeps = 0,
            initial_states = batch_three,
            initial_states_generator = gen,
            seed = 42,
        )
        MOI.set(model_trunc, QUBODrivers.FinalNumberOfReads(), 2)
        MOI.optimize!(model_trunc)
        counts_trunc = Dict(
            Tuple(Int(round(MOI.get(model_trunc, MOI.VariablePrimal(i), v))) for v in s_trunc) =>
                MOI.get(model_trunc, QUBOTools_MOI.NumberOfReads(i))
            for i in 1:MOI.get(model_trunc, MOI.ResultCount())
        )
        Test.@test counts_trunc == Dict(
            Tuple(batch_three[1, :]) => 1,
            Tuple(batch_three[2, :]) => 1,
        )
    end
end

Test.@testset "Neal initial_states validation errors" begin
    linear = [0.5, -0.75, 0.25]
    edges = [(1, 2), (2, 3)]
    quadratic = [1.0, -0.5]
    n = 4

    for bad_states in (
        [1, 0, 1],
        [1, 0, 1, 0, 1],
        zeros(Int, 0, 4),
        ones(Int, 2, 3),
        "invalid",
        123,
        [-1, 0, 1, 0],
        [2, 0, 1, 0],
        [0.5, 0.0, 1.0, 0.0],
        [1e30, 0.0, 1.0, 0.0],
        [NaN, 0.0, 1.0, 0.0],
    )
        model, _ = _wrapper_neal_qubo_model(
            linear,
            edges,
            quadratic,
            n;
            domain = :bool,
            num_reads = 2,
            num_sweeps = 1,
            initial_states = bad_states,
        )
        Test.@test_throws ArgumentError MOI.optimize!(model)
    end

    for bad_states in (
        [1, -1, 1],
        [1, -1, 1, -1, 1],
        zeros(Int, 0, 4),
        ones(Int, 2, 5),
        (1, -1, 1, -1),
        [0, 1, -1, 1],
        [2, 1, -1, 1],
        [true, false, true, false],
        [0.5, 1.0, -1.0, 1.0],
        [1e30, 1.0, -1.0, 1.0],
        [Inf, 1.0, -1.0, 1.0],
    )
        model, _ = _wrapper_neal_qubo_model(
            linear,
            edges,
            quadratic,
            n;
            domain = :spin,
            num_reads = 2,
            num_sweeps = 1,
            initial_states = bad_states,
        )
        Test.@test_throws ArgumentError MOI.optimize!(model)
    end
end
