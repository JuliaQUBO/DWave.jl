import DWave
import QUBODrivers
import QUBOTools
import Test

Test.@testset "Neal interrupted completed work regression" begin
    h = [0.5, -0.25]
    J = [0.0 -1.0; 0.0 0.0]
    calls = Ref(0)
    model, _ = _wrapper_neal_model(h, J;
        num_reads = 7, num_sweeps = 10, seed = 77,
        interrupt_function = () -> (calls[] += 1; true),
    )
    MOI.optimize!(model)
    raw = MOI.get(model, MOI.RawSolver())
    samples = QUBOTools.solution(raw)
    metadata = QUBOTools.metadata(samples)
    Test.@test calls[] == 1
    Test.@test sum(QUBOTools.reads(sample) for sample in samples) == 1
    Test.@test metadata["optimizer"]["evaluations"] == 1
    Test.@test MOI.get(model, MOI.TerminationStatus()) == MOI.INTERRUPTED
end

function _assert_neal_incumbents(model, variables, h, J)
    nresults = MOI.get(model, MOI.ResultCount())
    Test.@test nresults > 0
    for i in 1:nresults
        Test.@test MOI.get(model, MOI.PrimalStatus(i)) == MOI.FEASIBLE_POINT
        state = [MOI.get(model, MOI.VariablePrimal(i), v) for v in variables]
        Test.@test all(x -> x in (-1, 1), state)
        expected = sum(h .* state) + sum(J[i, j] * state[i] * state[j]
            for i in eachindex(h) for j in (i + 1):length(h))
        Test.@test MOI.get(model, MOI.ObjectiveValue(i)) ≈ expected
    end
    Test.@test MOI.get(model, MOI.PrimalStatus(nresults + 1)) == MOI.NO_SOLUTION
end

function _neal_metadata(model)
    return QUBOTools.metadata(QUBOTools.solution(MOI.get(model, MOI.RawSolver())))
end

function _with_mock_neal(f; occurrences = [4, 3], callback_args = ())
    original = DWave.Neal.dwave_samplers.SimulatedAnnealingSampler
    DWave.PythonCall.pyexec(
        """
def make_mock(dimod, np, occurrences, callback_args):
    class MockSampler:
        def sample(self, bqm, **params):
            callback = params['interrupt_function']
            if callback is not None:
                callback(*callback_args)
            states = [[1, -1], [-1, 1]][:len(occurrences)]
            return dimod.SampleSet.from_samples(
                (np.array(states, dtype=int).reshape((-1, 2)), [0, 1]), vartype='SPIN',
                energy=[bqm.energy(dict(enumerate(s))) for s in states],
                num_occurrences=occurrences,
                info={'effective_num_reads': params['num_reads']},
            )
    return MockSampler

target.SimulatedAnnealingSampler = make_mock(dimod, np, list(occurrences), tuple(callback_args))
""",
        @__MODULE__,
        (target = DWave.Neal.dwave_samplers, dimod = DWave.dwave_dimod, np = DWave.Neal.np,
            occurrences = occurrences, callback_args = callback_args),
    )
    try
        return f()
    finally
        _restore_neal_sampler!(original)
    end
end

Test.@testset "Neal deterministic read accounting and termination" begin
    h = [0.5, -0.25]
    J = [0.0 -1.0; 0.0 0.0]
    for (occurrences, callback, expected_status, reason) in (
        ([4, 3], nothing, MOI.LOCALLY_SOLVED, "completed"),
        ([4, 3], () -> false, MOI.LOCALLY_SOLVED, "completed"),
        ([4, 3], () -> true, MOI.INTERRUPTED, "interrupt_callback"),
        ([1], () -> true, MOI.INTERRUPTED, "interrupt_callback"),
        ([1], nothing, MOI.OTHER_ERROR, "incomplete_return"),
        (Int[], () -> true, MOI.INTERRUPTED, "interrupt_callback"),
        (Int[], nothing, MOI.OTHER_ERROR, "incomplete_return"),
    )
        _with_mock_neal(; occurrences) do
            model, variables = _wrapper_neal_model(h, J;
                num_reads = 11, interrupt_function = callback,
            )
            MOI.set(model, QUBODrivers.FinalNumberOfReads(), 7)
            MOI.optimize!(model)
            metadata = _neal_metadata(model)
            Test.@test isempty(QUBODrivers.validate_metadata(metadata))
            Test.@test metadata["reads"]["number_of_reads"] == 11
            Test.@test metadata["reads"]["final_number_of_reads"] == 7
            Test.@test metadata["dwave_info"]["effective_num_reads"] == 7
            Test.@test metadata["reads"]["completed_number_of_reads"] == sum(occurrences)
            Test.@test metadata["optimizer"]["evaluations"] == sum(occurrences)
            Test.@test metadata["optimizer"]["evaluation_unit"] == "returned_reads"
            Test.@test metadata["termination_status"] == expected_status
            Test.@test metadata["termination"]["reason"] == reason
            Test.@test metadata["termination"]["interrupt_requested"] == (expected_status == MOI.INTERRUPTED)
            Test.@test !metadata["termination"]["interrupt_callback_error"]
            Test.@test MOI.get(model, MOI.TerminationStatus()) == expected_status
            Test.@test MOI.get(model, MOI.RawStatusString()) ==
                (expected_status == MOI.LOCALLY_SOLVED ? "locally_solved" : reason)
            Test.@test MOI.get(model, MOI.ResultCount()) == length(occurrences)
            if isempty(occurrences)
                Test.@test MOI.get(model, MOI.PrimalStatus()) == MOI.NO_SOLUTION
            else
                _assert_neal_incumbents(model, variables, h, J)
            end
        end
    end

    Test.@testset "callback argument and truth conversion forwarding" begin
        calls = Int[]
        _with_mock_neal(; occurrences = [1], callback_args = (19,)) do
            model, _ = _wrapper_neal_model(h, J; num_reads = 7,
                interrupt_function = arg -> (push!(calls, arg); 1),
            )
            MOI.optimize!(model)
            Test.@test calls == [19]
            Test.@test MOI.get(model, MOI.TerminationStatus()) == MOI.INTERRUPTED
        end
    end

    Test.@testset "propagated callback exception clears prior solve" begin
        model, _ = _wrapper_neal_model(h, J; num_reads = 7, seed = 17, num_sweeps = 10,
            interrupt_function = () -> true,
        )
        MOI.optimize!(model)
        raw = MOI.get(model, MOI.RawSolver())
        Test.@test MOI.get(raw, MOI.TerminationStatus()) == MOI.INTERRUPTED
        calls = Ref(0)
        MOI.set(raw, MOI.RawOptimizerAttribute("interrupt_function"),
            () -> (calls[] += 1; error("callback sentinel")),
        )
        _with_mock_neal(; occurrences = [1]) do
            err = try
                MOI.optimize!(raw)
                nothing
            catch e
                e
            end
            Test.@test err isa NealPyException
            Test.@test occursin("callback sentinel", sprint(showerror, err))
            Test.@test calls[] == 1
            Test.@test MOI.get(raw, MOI.TerminationStatus()) == MOI.OTHER_ERROR
            Test.@test MOI.get(raw, MOI.RawStatusString()) == "sampling_error"
            Test.@test MOI.get(raw, MOI.ResultCount()) == 0
            Test.@test MOI.get(raw, MOI.PrimalStatus()) == MOI.NO_SOLUTION
        end
        MOI.set(raw, MOI.RawOptimizerAttribute("interrupt_function"), nothing)
        MOI.optimize!(raw)
        Test.@test MOI.get(raw, MOI.TerminationStatus()) == MOI.LOCALLY_SOLVED
        Test.@test !_neal_metadata(raw)["termination"]["interrupt_requested"]
    end
end

Test.@testset "Neal native callback path and solve isolation" begin
    h = [0.5, -0.25]
    J = [0.0 -1.0; 0.0 0.0]
    raw = DWave.Neal.Optimizer()
    Test.@test MOI.get(raw, MOI.TerminationStatus()) == MOI.OPTIMIZE_NOT_CALLED
    Test.@test MOI.get(raw, MOI.RawStatusString()) == ""
    Test.@test MOI.get(raw, MOI.ResultCount()) == 0
    Test.@test MOI.get(raw, MOI.PrimalStatus()) == MOI.NO_SOLUTION

    calls = Ref(0)
    model, variables = _wrapper_neal_model(h, J; num_reads = 11,
        num_sweeps = 10, seed = 77,
        interrupt_function = () -> (calls[] += 1; true),
    )
    MOI.set(model, QUBODrivers.FinalNumberOfReads(), 7)
    for (callback, count, completed, status, reason) in (
        (() -> (calls[] += 1; true), 1, 1, MOI.INTERRUPTED, "interrupt_callback"),
        (nothing, 0, 7, MOI.LOCALLY_SOLVED, "completed"),
        (() -> (calls[] += 1; false), 7, 7, MOI.LOCALLY_SOLVED, "completed"),
        (() -> (calls[] += 1; calls[] == 7), 7, 7, MOI.INTERRUPTED, "interrupt_callback"),
        (() -> (calls[] += 1; error("native callback sentinel")), 1, 1, MOI.OTHER_ERROR, "interrupt_callback_error"),
        (nothing, 0, 7, MOI.LOCALLY_SOLVED, "completed"),
    )
        calls[] = 0
        MOI.set(model, MOI.RawOptimizerAttribute("interrupt_function"), callback)
        MOI.optimize!(model)
        metadata = _neal_metadata(model)
        Test.@test calls[] == count
        Test.@test metadata["reads"]["number_of_reads"] == 11
        Test.@test metadata["reads"]["final_number_of_reads"] == 7
        Test.@test metadata["reads"]["completed_number_of_reads"] == completed
        Test.@test metadata["optimizer"]["evaluations"] == completed
        Test.@test MOI.get(model, MOI.TerminationStatus()) == status
        Test.@test metadata["termination"]["reason"] == reason
        Test.@test MOI.get(model, MOI.RawStatusString()) ==
            (status == MOI.LOCALLY_SOLVED ? "locally_solved" : reason)
        Test.@test metadata["termination"]["interrupt_requested"] == (reason == "interrupt_callback")
        Test.@test metadata["termination"]["interrupt_callback_error"] == (reason == "interrupt_callback_error")
        _assert_neal_incumbents(model, variables, h, J)
    end
    raw = MOI.get(model, MOI.RawSolver())
    MOI.empty!(raw)
    Test.@test MOI.get(raw, MOI.TerminationStatus()) == MOI.OPTIMIZE_NOT_CALLED
    Test.@test MOI.get(raw, MOI.RawStatusString()) == ""
    Test.@test MOI.get(raw, MOI.ResultCount()) == 0
end
