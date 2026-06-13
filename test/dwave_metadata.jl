import DWave
import QUBODrivers
import QUBOTools
import Test

const MOI = QUBODrivers.MOI

function _fake_embedding_sampler()
    ans = DWave.PythonCall.pyexec(
        @NamedTuple{sampler::DWave.PythonCall.Py},
        """
class FakeEmbeddingSampler:
    def __init__(self):
        self.child = type("FakeChildSampler", (), {})()
        self.child.properties = {
            "chip_id": "mock-chip",
            "topology": {"type": "zephyr", "shape": [4, 4]},
            "category": "qpu",
        }
        self.child.solver = type("FakeSolver", (), {"name": "Advantage_system6.4"})()

    def sample_ising(self, h, J, **params):
        return type(
            "FakeSampleSet",
            (),
            {
                "variables": [1, 2],
                "record": [([1, -1], -1.25, 3)],
                "info": {
                    "problem_id": "mock-problem",
                    "timing": {"qpu_access_time": 42},
                },
            },
        )()

sampler = FakeEmbeddingSampler()
""",
        @__MODULE__,
        (),
    )

    return ans.sampler
end

function _wrapper_dwave_sampleset(sampler)
    model = MOI.instantiate(DWave.Optimizer; with_bridge_type = Float64)
    variables, _ = MOI.add_constrained_variables(model, fill(QUBODrivers.Spin(), 2))

    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(
        model,
        MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(),
        MOI.ScalarQuadraticFunction{Float64}(
            MOI.ScalarQuadraticTerm{Float64}[
                MOI.ScalarQuadraticTerm{Float64}(-1.0, variables[1], variables[2]),
            ],
            MOI.ScalarAffineTerm{Float64}[
                MOI.ScalarAffineTerm{Float64}(0.5, variables[1]),
                MOI.ScalarAffineTerm{Float64}(-0.25, variables[2]),
            ],
            0.0,
        ),
    )
    MOI.set(model, MOI.RawOptimizerAttribute("sampler"), sampler)

    MOI.optimize!(model)

    return QUBOTools.solution(MOI.get(model, MOI.RawSolver()))
end

Test.@testset "DWave metadata includes chip info" begin
    metadata = QUBOTools.metadata(_wrapper_dwave_sampleset(_fake_embedding_sampler()))

    Test.@test haskey(metadata, "dwave_info")
    Test.@test isempty(QUBODrivers.validate_metadata(metadata))
    Test.@test metadata["algorithm"]["name"] == "D-Wave Quantum Annealing Sampler"
    Test.@test metadata["backend"]["name"] == "dwave-ocean-sdk"
    Test.@test metadata["backend"]["version"] == DWave.OCEAN_SDK_VERSION
    Test.@test metadata["reads"]["number_of_reads"] == 100
    Test.@test metadata["reads"]["final_number_of_reads"] == 100
    Test.@test metadata["time"]["effective"] == 42 / 1_000_000
    Test.@test metadata["time"]["dwave"]["timing"]["qpu_access_time"] == 42
    Test.@test metadata["time"]["dwave"]["units"]["qpu_access_time"] == "microseconds"

    info = metadata["dwave_info"]
    Test.@test info["problem_id"] == "mock-problem"
    Test.@test info["timing"]["qpu_access_time"] == 42
    Test.@test haskey(info, "chip_info")

    chip_info = info["chip_info"]
    Test.@test chip_info["chip_id"] == "mock-chip"
    Test.@test chip_info["category"] == "qpu"
    Test.@test chip_info["solver_name"] == "Advantage_system6.4"
    Test.@test chip_info["topology"]["type"] == "zephyr"
    Test.@test chip_info["topology"]["shape"] == Any[4, 4]
end
