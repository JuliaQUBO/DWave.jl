import DWave
import QUBODrivers
import QUBOTools
import Test

# Capture the actual native boundary, with either exhaustive returned states or
# the real upstream annealer. Exhaustive energies exercise public reconstruction
# without requiring a stochastic optimizer to find an optimum.
function _with_neal_payload(f; exhaustive = true)
    original = DWave.Neal.dwave_samplers.SimulatedAnnealingSampler
    DWave.PythonCall.pyexec(
        """
def capture_sampler(original, dimod, np, itertools, exhaustive):
    class CaptureSampler:
        last = None
        def sample_ising(self, *args, **params):
            return original().sample_ising(*args, **params)
        def sample(self, bqm, **params):
            CaptureSampler.last = (bqm.copy(), dict(params))
            if not exhaustive:
                return original().sample(bqm, **params)
            states = np.array(list(itertools.product((-1, 1), repeat=bqm.num_variables)), dtype=np.int8)
            return dimod.SampleSet.from_samples_bqm((states, list(bqm.variables)), bqm)
    return CaptureSampler

target.SimulatedAnnealingSampler = capture_sampler(original, dimod, np, itertools, exhaustive)
""",
        @__MODULE__,
        (target = DWave.Neal.dwave_samplers, original = original,
            dimod = DWave.dwave_dimod, np = DWave.Neal.np,
            itertools = DWave.PythonCall.pyimport("itertools"), exhaustive = exhaustive),
    )
    try
        return f()
    finally
        _restore_neal_sampler!(original)
    end
end

function _scale_fixture(domain, sense, scale; constant = false, decimal = false, offset = 2.25)
    # Variable 4 is isolated; exactly representable coefficients make seeded
    # trajectory equality meaningful. The decimal fixture retains issue #74.
    linear = constant ? zeros(3) : decimal ? [0.3, -0.7, 0.2] : [0.5, -0.75, 0.25]
    quadratic = constant ? zeros(2) : decimal ? [1.2, -0.9] : [1.0, -0.5]
    variables = MOI.VariableIndex.(1:4)
    model = QUBOTools.Model{MOI.VariableIndex,Float64,Int}(
        variables, [1, 2, 3], linear, [1, 2], [2, 3], quadratic;
        domain, sense, scale, offset,
    )
    # This MOI source absorbs scale into the original objective coefficients.
    # It is built independently of ising(), matching the actual copy_to route.
    source = MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}())
    vars, _ = MOI.add_constrained_variables(source,
        fill(domain == :bool ? MOI.ZeroOne() : QUBODrivers.Spin(), 4))
    MOI.set(source, MOI.ObjectiveSense(), sense == :min ? MOI.MIN_SENSE : MOI.MAX_SENSE)
    MOI.set(source, MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(),
        MOI.ScalarQuadraticFunction(
            [MOI.ScalarQuadraticTerm(scale * quadratic[k], vars[k], vars[k + 1]) for k in 1:2],
            [MOI.ScalarAffineTerm(scale * linear[k], vars[k]) for k in 1:3],
            scale * offset,
        ))
    energy = x -> scale * (sum(linear .* x[1:3]) +
        quadratic[1] * x[1] * x[2] + quadratic[2] * x[2] * x[3] + offset)
    return model, source, energy
end

function _scale_optimizer(model, source, route; kwargs...)
    optimizer = DWave.Neal.Optimizer()
    if route == :direct
        QUBODrivers.set_model!(optimizer, model)
    else
        MOI.copy_to(optimizer, source)
    end
    for (name, value) in pairs(kwargs)
        MOI.set(optimizer, MOI.RawOptimizerAttribute(String(name)), value)
    end
    return optimizer
end

function _native_payload()
    bqm, params = DWave.Neal.dwave_samplers.SimulatedAnnealingSampler.last
    variables = DWave.PythonCall.pyconvert(Vector{Int}, DWave.PythonCall.pybuiltins.list(bqm.variables))
    vectors = bqm.to_numpy_vectors(variable_order = variables)
    return (bqm = bqm, params = params, variables = variables,
        linear = DWave.PythonCall.pyconvert(Vector{Float64}, vectors.linear_biases),
        rows = DWave.PythonCall.pyconvert(Vector{Int}, vectors.quadratic.row_indices),
        cols = DWave.PythonCall.pyconvert(Vector{Int}, vectors.quadratic.col_indices),
        weights = DWave.PythonCall.pyconvert(Vector{Float64}, vectors.quadratic.biases),
        offset = DWave.PythonCall.pyconvert(Float64, vectors.offset))
end

_native_energy(payload, s) = DWave.PythonCall.pyconvert(Float64,
    payload.bqm.energy(Dict(i - 1 => s[i] for i in eachindex(s))))

Test.@testset "Neal scale: native energy units and public reconstruction" begin
    _with_neal_payload() do
        for domain in (:bool, :spin), sense in (:min, :max), scale in (1.0, 1.5, -1.5, 0.0),
            constant in (false, true)
            Test.@testset "$domain/$sense/scale=$scale/constant=$constant" begin
                model, source, energy = _scale_fixture(domain, sense, scale; constant)
                original_form = deepcopy(Tuple(QUBOTools.ising(model, :dict)))
                sign = sense == :min ? 1 : -1
                payloads = []
                for route in (:direct, :moi)
                    optimizer = _scale_optimizer(model, source, route;
                        num_reads = 16, num_sweeps = 6, beta_range = (0.25, 2.0), seed = 41)
                    MOI.optimize!(optimizer)
                    payload = _native_payload()
                    push!(payloads, payload)
                    Test.@test payload.variables == collect(0:3)
                    Test.@test payload.offset == 0.0
                    Test.@test DWave.PythonCall.pyconvert(Tuple{Float64,Float64}, payload.params["beta_range"]) == (0.25, 2.0)
                    Test.@test MOI.get(optimizer, MOI.TerminationStatus()) == MOI.LOCALLY_SOLVED
                    Test.@test QUBOTools.metadata(QUBOTools.solution(optimizer))["reads"]["completed_number_of_reads"] == 16
                    reference_s = [-1, -1, -1, -1]
                    reference_x = domain == :bool ? [0, 0, 0, 0] : reference_s
                    reference_e = _native_energy(payload, reference_s)
                    for (result_index, sample) in enumerate(QUBOTools.solution(optimizer))
                        x = QUBOTools.state(sample)
                        s = domain == :bool ? 2 .* x .- 1 : x
                        Test.@test QUBOTools.value(sample) ≈ energy(x) atol = 1e-14
                        Test.@test MOI.get(optimizer, MOI.ObjectiveValue(result_index)) ≈ energy(x) atol = 1e-14
                        Test.@test [MOI.get(optimizer, MOI.VariablePrimal(result_index), v)
                            for v in MOI.VariableIndex.(1:4)] == x
                        Test.@test QUBOTools.value(x, QUBOTools.form(model)) ≈ energy(x) atol = 1e-14
                        delta = _native_energy(payload, s) - reference_e
                        Test.@test delta ≈ sign * (energy(x) - energy(reference_x)) atol = 1e-14
                        for beta in (0.25, 2.0)
                            Test.@test beta * delta ≈ beta * sign * (energy(x) - energy(reference_x)) atol = 2e-14
                        end
                    end
                end
                a, b = payloads
                Test.@test a.linear == b.linear
                Test.@test a.rows == b.rows
                Test.@test a.cols == b.cols
                Test.@test a.weights == b.weights
                # Conversion and sampling leave the caller's model untouched.
                Test.@test Tuple(QUBOTools.ising(model, :dict)) == original_form
            end
        end
    end
end

Test.@testset "Neal scale: retained decimal payload" begin
    _with_neal_payload() do
        model, source, _ = _scale_fixture(:bool, :min, 1.5; decimal = true)
        payloads = []
        for route in (:direct, :moi)
            MOI.optimize!(_scale_optimizer(model, source, route; num_reads = 16,
                beta_schedule_type = "custom", beta_schedule = [0.125, 0.5, 2.0]))
            payload = _native_payload()
            push!(payloads, payload)
            Test.@test maximum(abs, payload.linear) ≈ 0.675
            Test.@test maximum(abs, payload.weights) ≈ 0.45
            Test.@test DWave.PythonCall.pyconvert(Vector{Float64}, payload.params["beta_schedule"]) == [0.125, 0.5, 2.0]
        end
        # Different floating-point evaluation orders can differ by a few ulps.
        Test.@test payloads[1].linear ≈ payloads[2].linear atol = 2e-16 rtol = 0
        Test.@test payloads[1].weights ≈ payloads[2].weights atol = 2e-16 rtol = 0
    end
end

function _scale_records(optimizer)
    return sort([(Tuple(QUBOTools.state(sample)), QUBOTools.value(sample), QUBOTools.reads(sample))
        for sample in QUBOTools.solution(optimizer)])
end

# Independent translation of the original fixture, not the wrapper's ising().
function _expected_native(domain, sense, scale; constant = false)
    l = constant ? zeros(3) : [0.5, -0.75, 0.25]
    q = constant ? zeros(2) : [1.0, -0.5]
    factor = (sense == :min ? 1 : -1) * scale
    h = domain == :spin ? [l; 0.0] :
        [l[1] / 2 + q[1] / 4, l[2] / 2 + sum(q) / 4, l[3] / 2 + q[2] / 4, 0.0]
    J = zeros(4, 4)
    J[1, 2], J[2, 3] = domain == :spin ? q : q ./ 4
    return factor .* h, factor .* J
end

# Resolved dwave-samplers 1.7.0 _default_ising_beta_range: hot gap uses
# the largest sum of incident absolute biases; cold gap uses the smallest
# nonzero incident bias and the number of spins sharing it (excitation=0.01).
function _expected_auto_range(h, J)
    sums = abs.(h)
    minima = [iszero(v) ? Inf : abs(v) for v in h]
    for i in eachindex(h), j in (i + 1):length(h)
        v = abs(J[i, j])
        sums[i] += v
        sums[j] += v
        if !iszero(v)
            minima[i] = min(minima[i], v)
            minima[j] = min(minima[j], v)
        end
    end
    smallest = minimum(minima)
    isinf(smallest) && return [0.1, 1.0]
    return [log(2) / (2 * maximum(sums)), log(count(==(smallest), minima) / 0.01) / (2 * smallest)]
end

Test.@testset "Neal scale: schedules and seeded native parity" begin
    schedules = (
        (; beta_schedule_type = "linear", beta_range = (0.125, 2.0)),
        (; beta_schedule_type = "geometric", beta_range = (0.125, 2.0)),
        (; beta_schedule_type = "custom", beta_schedule = [0.125, 0.5, 2.0]),
        (; beta_schedule_type = "linear"),
        (; beta_schedule_type = "geometric"),
    )
    _with_neal_payload(; exhaustive = false) do
        for domain in (:bool, :spin), sense in (:min, :max), scale in (1.0, 1.5, -1.5, 0.0),
            constant in (false, true), schedule in schedules
            Test.@testset "$domain/$sense/$scale/constant=$constant/$schedule" begin
                model, source, energy = _scale_fixture(domain, sense, scale; constant)
                h, J = _expected_native(domain, sense, scale; constant)
                sweeps = haskey(schedule, :beta_schedule) ? 2 * length(schedule.beta_schedule) : 12
                kwargs = (; num_reads = 24, num_sweeps = sweeps, num_sweeps_per_beta = 2,
                    seed = 202_610_11, schedule...)
                # Direct upstream execution on independently derived coefficients.
                native = _direct_neal_records(h, J; kwargs...)
                expected = sort([(Tuple(domain == :bool ? (collect(r.state) .+ 1) .÷ 2 : r.state),
                    energy(domain == :bool ? (collect(r.state) .+ 1) .÷ 2 : collect(r.state)), r.reads)
                    for r in _aggregate_records(native)])
                route_records = []
                for route in (:direct, :moi)
                    optimizer = _scale_optimizer(model, source, route; kwargs...)
                    MOI.optimize!(optimizer)
                    payload = _native_payload()
                    Test.@test payload.variables == collect(0:3)
                    Test.@test payload.linear == h
                    params = payload.params
                    Test.@test DWave.PythonCall.pyconvert(Int, params["seed"]) == kwargs.seed
                    Test.@test DWave.PythonCall.pyconvert(Int, params["num_sweeps_per_beta"]) == 2
                    Test.@test DWave.PythonCall.pyconvert(String, params["beta_schedule_type"]) == schedule.beta_schedule_type
                    info = QUBOTools.metadata(QUBOTools.solution(optimizer))
                    Test.@test info["reads"]["completed_number_of_reads"] == 24
                    Test.@test MOI.get(optimizer, MOI.TerminationStatus()) == MOI.LOCALLY_SOLVED
                    if haskey(schedule, :beta_schedule)
                        Test.@test DWave.PythonCall.pyconvert(Vector{Float64}, params["beta_schedule"]) == schedule.beta_schedule
                        Test.@test info["dwave_info"]["beta_range"] === nothing
                    else
                        expected_range = haskey(schedule, :beta_range) ? collect(schedule.beta_range) : _expected_auto_range(h, J)
                        Test.@test info["dwave_info"]["beta_range"] ≈ expected_range
                        if haskey(schedule, :beta_range)
                            Test.@test DWave.PythonCall.pyconvert(Tuple{Float64,Float64}, params["beta_range"]) == schedule.beta_range
                        else
                            Test.@test DWave.PythonCall.pyis(params["beta_range"], DWave.PythonCall.pybuiltins.None)
                            if !constant && !iszero(scale)
                                unit_h, unit_J = _expected_native(domain, sense, 1.0)
                                Test.@test expected_range ≈ _expected_auto_range(unit_h, unit_J) ./ abs(scale)
                            end
                        end
                    end
                    records = _scale_records(optimizer)
                    push!(route_records, records)
                    Test.@test records == expected
                    Test.@test sum(last, records) == 24
                end
                Test.@test route_records[1] == route_records[2]
            end
        end
    end
end


Test.@testset "Neal offsets leave native transitions unchanged" begin
    _with_neal_payload(; exhaustive = false) do
        for domain in (:bool, :spin), sense in (:min, :max), route in (:direct, :moi)
            outcomes = []
            for offset in (2.25, -4.0)
                model, source, energy = _scale_fixture(domain, sense, -1.5; offset)
                optimizer = _scale_optimizer(model, source, route; num_reads = 24,
                    num_sweeps = 12, beta_range = (0.125, 2.0), seed = 77)
                MOI.optimize!(optimizer)
                payload = _native_payload()
                records = _scale_records(optimizer)
                Test.@test all(r -> r[2] == energy(collect(r[1])), records)
                push!(outcomes, (payload, records))
            end
            (a, ar), (b, br) = outcomes
            Test.@test a.linear == b.linear
            Test.@test a.weights == b.weights
            Test.@test a.offset == b.offset == 0.0
            Test.@test [(r[1], r[3]) for r in ar] == [(r[1], r[3]) for r in br]
            Test.@test [r[2] for r in br] == [r[2] + (-1.5) * (-4.0 - 2.25) for r in ar]
        end
    end
end
