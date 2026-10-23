import DWave
import QUBODrivers
import QUBOTools
import Test

const MOI = QUBODrivers.MOI
const AUTH_PACKAGE_ROOT = normpath(joinpath(@__DIR__, ".."))

function _install_cloud_spies!(; fail_solver::Bool = false)
    return DWave.PythonCall.pyexec(
        @NamedTuple{
            state::DWave.PythonCall.Py,
            restore::DWave.PythonCall.Py,
        },
        """
def make_spies(dwave_cloud, dwave_system, fail_solver):
    orig_client = getattr(dwave_cloud, "Client", None)
    orig_sampler = getattr(dwave_system, "DWaveSampler", None)
    orig_composite = getattr(dwave_system, "EmbeddingComposite", None)

    state = {
        "client_tokens": [],
        "client_endpoints": [],
        "client_types": [],
        "get_solver_calls": 0,
        "client_closed": 0,
        "sampler_tokens": [],
        "sampler_endpoints": [],
        "sampler_client_types": [],
        "composite_calls": 0,
        "sample_ising_calls": 0,
    }

    class FakeCloudClient:
        @classmethod
        def from_config(cls, config_file=None, profile=None, client=None, **kwargs):
            config = dwave_cloud.config.load_config(
                config_file=config_file,
                profile=profile,
                client=client,
                **kwargs,
            )
            client_type = config.pop("client", None) or "base"
            return cls(_client_type=client_type, **config)

        def __init__(self, *args, **kwargs):
            client_type = kwargs.pop("_client_type", "base")
            raw_token = kwargs.get("token")
            token = raw_token.strip() if isinstance(raw_token, str) else None
            if not token:
                raise ValueError("API token not defined")
            endpoint = kwargs.get("endpoint")
            self.client_type = client_type
            state["client_tokens"].append(token)
            state["client_endpoints"].append(endpoint)
            state["client_types"].append(client_type)
            self.config = type(
                "FakeConfig",
                (),
                {"token": token, "endpoint": endpoint, "client": client_type},
            )()

        def get_solver(self, *args, **kwargs):
            state["get_solver_calls"] += 1
            if self.client_type != "qpu":
                raise ValueError(
                    f"Solver with the requested features not available for client={self.client_type}"
                )
            if fail_solver:
                raise ValueError("Invalid token")
            return {"id": "mock-solver", "category": "qpu"}

        def close(self):
            state["client_closed"] += 1

    class FakeDWaveSampler:
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("client", "qpu")
            config = dwave_cloud.config.load_config(**kwargs)
            raw_token = config.get("token")
            token = raw_token.strip() if isinstance(raw_token, str) else None
            if not token:
                raise ValueError("API token not defined")
            client_type = config.get("client") or "base"
            if client_type != "qpu":
                raise ValueError(
                    f"Solver with the requested features not available for client={client_type}"
                )
            endpoint = config.get("endpoint")
            state["sampler_tokens"].append(token)
            state["sampler_endpoints"].append(endpoint)
            state["sampler_client_types"].append(client_type)
            self.properties = {
                "chip_id": "mock-chip",
                "topology": {"type": "pegasus", "shape": [16]},
                "category": "qpu",
                "qubits": [0, 1],
                "couplers": [[0, 1]],
                "num_qubits": 2,
            }
            self.solver = type("FakeSolver", (), {"name": "mock-solver"})()

    class FakeEmbeddingComposite:
        def __init__(self, child):
            state["composite_calls"] += 1
            self.child = child

        def sample_ising(self, h, J, **params):
            state["sample_ising_calls"] += 1
            return type(
                "FakeSampleSet",
                (),
                {
                    "variables": [1, 2],
                    "record": [([1, -1], -1.0, params.get("num_reads", 1))],
                    "info": {
                        "problem_id": "mock-problem",
                        "timing": {"qpu_access_time": 10},
                    },
                },
            )()

    dwave_cloud.Client = FakeCloudClient
    dwave_system.DWaveSampler = FakeDWaveSampler
    dwave_system.EmbeddingComposite = FakeEmbeddingComposite

    def restore():
        if orig_client is not None:
            dwave_cloud.Client = orig_client
        if orig_sampler is not None:
            dwave_system.DWaveSampler = orig_sampler
        if orig_composite is not None:
            dwave_system.EmbeddingComposite = orig_composite

    return state, restore

state, restore = make_spies(dwave_cloud, dwave_system, fail_solver)
""",
        @__MODULE__,
        (
            dwave_cloud = DWave.dwave_cloud,
            dwave_system = DWave.dwave_system,
            fail_solver = fail_solver,
        ),
    )
end

function _with_in_memory_dwave_config(f::Function, contents::AbstractString)
    ans = DWave.PythonCall.pyexec(
        @NamedTuple{patcher::DWave.PythonCall.Py},
        """
from unittest.mock import patch, mock_open
patcher = patch("dwave.cloud.config.loaders.open", mock_open(read_data=contents), create=True)
patcher.start()
""",
        @__MODULE__,
        (contents = String(contents),),
    )
    try
        virtual_path = "/virtual/dwave-auth-test.conf"
        return withenv("DWAVE_CONFIG_FILE" => virtual_path) do
            return f(virtual_path)
        end
    finally
        ans.patcher.stop()
    end
end

function _with_temp_dwave_config(f::Function, contents::AbstractString)
    temp_file = try
        mktemp()
    catch
        nothing
    end
    temp_file === nothing && return _with_in_memory_dwave_config(f, contents)

    path, io = temp_file
    try
        write(io, contents)
        close(io)
        return withenv("DWAVE_CONFIG_FILE" => path) do
            return f(path)
        end
    finally
        try
            isfile(path) && rm(path; force = true)
        catch
        end
    end
end

function _with_cloud_spies(f::Function; fail_solver::Bool = false)
    spies = _install_cloud_spies!(; fail_solver = fail_solver)
    prev_token = DWave.API_TOKEN[]
    try
        return _with_temp_dwave_config("") do _
            withenv(
                "DWAVE_PROFILE" => nothing,
                "DWAVE_API_CLIENT" => nothing,
                "DWAVE_API_ENDPOINT" => nothing,
                "DWAVE_API_REGION" => nothing,
            ) do
                return f(spies.state)
            end
        end
    finally
        spies.restore()
        DWave.API_TOKEN[] = prev_token
    end
end

function _build_two_spin_cloud_model()
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
            MOI.ScalarAffineTerm{Float64}[],
            0.0,
        ),
    )
    MOI.set(model, DWave.NumberOfReads(), 5)

    return model
end

function _subprocess_import_check(initial_token::Union{String, Nothing})
    code = """
    const PythonCall = Base.require(
        Base.PkgId(Base.UUID("6099a3de-0909-46bc-b1f4-468b9a2dfc0d"), "PythonCall"),
    )
    dwave_cloud = PythonCall.pyimport("dwave.cloud")
    dwave_system = PythonCall.pyimport("dwave.system")

    ans = PythonCall.pyexec(
        @NamedTuple{counts::PythonCall.Py},
        \"\"\"
    def install_import_guards(dwave_cloud, dwave_system):
        counts = {"client": 0, "sampler": 0}
        class FailIfCalledClient:
            @classmethod
            def from_config(cls, *args, **kwargs):
                counts["client"] += 1
                raise RuntimeError("dwave.cloud.Client.from_config should not be called on import")
            def __init__(self, *args, **kwargs):
                counts["client"] += 1
                raise RuntimeError("dwave.cloud.Client should not be called on import")
        class FailIfCalledSampler:
            def __init__(self, *args, **kwargs):
                counts["sampler"] += 1
                raise RuntimeError("dwave.system.DWaveSampler should not be called on import")
        dwave_cloud.Client = FailIfCalledClient
        dwave_system.DWaveSampler = FailIfCalledSampler
        return counts
    counts = install_import_guards(dwave_cloud, dwave_system)
    \"\"\",
        Main,
        (dwave_cloud = dwave_cloud, dwave_system = dwave_system),
    )

    using DWave
    @assert DWave.API_TOKEN[] === nothing

    for token_val in (nothing, "", "   ", "unvalidated-host-token")
        DWave.API_TOKEN[] = "stale-token"
        if token_val === nothing
            delete!(ENV, "DWAVE_API_TOKEN")
        else
            ENV["DWAVE_API_TOKEN"] = token_val
        end
        DWave.__init__()
        @assert DWave.API_TOKEN[] === nothing
    end

    # Constructing local and cloud optimizers and running a local solve must not trigger cloud calls.
    neal = DWave.Neal.Optimizer()
    greedy = DWave.Greedy.Optimizer()
    random_opt = DWave.Random.Optimizer()
    tabu = DWave.Tabu.Optimizer()
    cloud_opt = DWave.Optimizer()

    model = DWave.QUBODrivers.MOI.instantiate(DWave.Neal.Optimizer; with_bridge_type = Float64)
    vars, _ = DWave.QUBODrivers.MOI.add_constrained_variables(model, fill(DWave.QUBODrivers.Spin(), 2))
    DWave.QUBODrivers.MOI.set(model, DWave.QUBODrivers.MOI.ObjectiveSense(), DWave.QUBODrivers.MOI.MIN_SENSE)
    DWave.QUBODrivers.MOI.set(
        model,
        DWave.QUBODrivers.MOI.ObjectiveFunction{DWave.QUBODrivers.MOI.ScalarQuadraticFunction{Float64}}(),
        DWave.QUBODrivers.MOI.ScalarQuadraticFunction{Float64}(
            [DWave.QUBODrivers.MOI.ScalarQuadraticTerm{Float64}(-1.0, vars[1], vars[2])],
            DWave.QUBODrivers.MOI.ScalarAffineTerm{Float64}[],
            0.0,
        ),
    )
    DWave.QUBODrivers.MOI.set(model, DWave.QUBODrivers.MOI.RawOptimizerAttribute("num_reads"), 4)
    DWave.QUBODrivers.MOI.set(model, DWave.QUBODrivers.MOI.RawOptimizerAttribute("num_sweeps"), 5)
    DWave.QUBODrivers.MOI.optimize!(model)

    counts = DWave.jl_object(ans.counts)
    @assert counts["client"] == 0
    @assert counts["sampler"] == 0
    """
    julia = joinpath(Sys.BINDIR, Base.julia_exename())
    active_proj = Base.active_project()
    project_dir = active_proj === nothing ? AUTH_PACKAGE_ROOT : dirname(active_proj)
    err_buf = IOBuffer()
    py_exe = String(DWave.PythonCall.C.CTX.exe_path)
    py_lib = String(DWave.PythonCall.C.CTX.lib_path)
    ok = withenv(
        "DWAVE_API_TOKEN" => initial_token,
        "JULIA_CONDAPKG_BACKEND" => "Null",
        "JULIA_PYTHONCALL_EXE" => py_exe,
        "JULIA_PYTHONCALL_LIB" => py_lib,
    ) do
        success(pipeline(
            `$julia --startup-file=no --project=$project_dir -e $code`,
            stderr = err_buf,
        ))
    end
    return ok, String(take!(err_buf))
end

Test.@testset "Local-only DWave import and classical setup make zero cloud calls" begin
    for initial_token in (nothing, "unvalidated-host-token")
        ok, err_output = _subprocess_import_check(initial_token)
        Test.@test ok
        Test.@test !occursin("DWAVE_API_TOKEN", err_output)
    end
end

Test.@testset "Explicit __auth__ validates credentials and reports diagnostics" begin
    _with_cloud_spies(; fail_solver = false) do state
        DWave.API_TOKEN[] = "stale-token"
        withenv("DWAVE_API_TOKEN" => "   ") do
            Test.@test !DWave.__auth__(; verbose = false)
            Test.@test DWave.API_TOKEN[] === nothing
            Test.@test (
                Test.@test_logs (
                    :warn,
                    r"The 'DWAVE_API_TOKEN' environment variable is not defined or is empty\.",
                ) !DWave.__auth__(; verbose = true)
            )
        end
        counts = DWave.jl_object(state)
        Test.@test isempty(counts["client_tokens"])
        Test.@test counts["get_solver_calls"] == 0
    end

    _with_cloud_spies(; fail_solver = true) do state
        withenv("DWAVE_API_TOKEN" => "invalid-token") do
            Test.@test !DWave.__auth__(; verbose = false)
            Test.@test DWave.API_TOKEN[] === nothing
            Test.@test (
                Test.@test_logs (
                    :warn,
                    r"The 'DWAVE_API_TOKEN' environment variable defined, but the token is not valid\.",
                ) !DWave.__auth__(; verbose = true)
            )
            Test.@test DWave.API_TOKEN[] === nothing
        end
        counts = DWave.jl_object(state)
        Test.@test counts["client_tokens"] == Any["invalid-token", "invalid-token"]
        Test.@test counts["client_types"] == Any["qpu", "qpu"]
        Test.@test counts["get_solver_calls"] == 2
        Test.@test counts["client_closed"] == 2
    end

    _with_cloud_spies(; fail_solver = false) do state
        withenv("DWAVE_API_TOKEN" => "  valid-token  ") do
            Test.@test DWave.__auth__(; verbose = false)
            Test.@test DWave.API_TOKEN[] == "valid-token"
        end
        counts = DWave.jl_object(state)
        Test.@test counts["client_tokens"] == Any["valid-token"]
        Test.@test counts["client_types"] == Any["qpu"]
        Test.@test counts["get_solver_calls"] == 1
        Test.@test counts["client_closed"] == 1
    end

    _with_cloud_spies(; fail_solver = false) do state
        config_text = """
        [defaults]
        token = config-only-token
        endpoint = https://configured.example.invalid/sapi/
        client = sw
        """
        _with_temp_dwave_config(config_text) do _
            for env_val in (nothing, "", "   ")
                DWave.API_TOKEN[] = nothing
                withenv("DWAVE_API_TOKEN" => env_val) do
                    Test.@test DWave.__auth__(; verbose = false)
                    Test.@test DWave.API_TOKEN[] == "config-only-token"
                end
            end
        end
        counts = DWave.jl_object(state)
        Test.@test counts["client_tokens"] == Any[
            "config-only-token",
            "config-only-token",
            "config-only-token",
        ]
        Test.@test counts["client_endpoints"] == Any[
            "https://configured.example.invalid/sapi/",
            "https://configured.example.invalid/sapi/",
            "https://configured.example.invalid/sapi/",
        ]
        Test.@test counts["client_types"] == Any["qpu", "qpu", "qpu"]
        Test.@test counts["get_solver_calls"] == 3
        Test.@test counts["client_closed"] == 3
    end
end

Test.@testset "Cloud DWave.Optimizer defers authentication until solve and reports diagnostics" begin
    _with_cloud_spies(; fail_solver = false) do state
        withenv("DWAVE_API_TOKEN" => "") do
            model = _build_two_spin_cloud_model()
            counts_before = DWave.jl_object(state)
            Test.@test isempty(counts_before["client_tokens"])
            Test.@test isempty(counts_before["sampler_tokens"])

            err = try
                MOI.optimize!(model)
                nothing
            catch e
                e
            end

            Test.@test err isa ArgumentError
            msg = sprint(showerror, err)
            Test.@test occursin("The 'DWAVE_API_TOKEN' environment variable is not defined or is empty.", msg)
            Test.@test occursin("No D-Wave Ocean configuration token was found", msg)
            Test.@test occursin("https://docs.ocean.dwavesys.com/en/stable/overview/sapi.html", msg)

            counts_after = DWave.jl_object(state)
            Test.@test isempty(counts_after["client_tokens"])
            Test.@test isempty(counts_after["sampler_tokens"])
        end
    end

    _with_cloud_spies(; fail_solver = true) do state
        withenv("DWAVE_API_TOKEN" => "bad-cloud-token") do
            model = _build_two_spin_cloud_model()
            err = try
                MOI.optimize!(model)
                nothing
            catch e
                e
            end

            Test.@test err isa ArgumentError
            msg = sprint(showerror, err)
            Test.@test occursin("The 'DWAVE_API_TOKEN' environment variable defined, but the token is not valid.", msg)
            Test.@test occursin("Invalid token", msg)
            Test.@test occursin("https://docs.ocean.dwavesys.com/en/stable/overview/sapi.html", msg)
            Test.@test DWave.API_TOKEN[] === nothing

            counts_after = DWave.jl_object(state)
            Test.@test counts_after["client_tokens"] == Any["bad-cloud-token"]
            Test.@test counts_after["client_types"] == Any["qpu"]
            Test.@test counts_after["get_solver_calls"] == 1
            Test.@test counts_after["client_closed"] == 1
            Test.@test isempty(counts_after["sampler_tokens"])
        end
    end

    _with_cloud_spies(; fail_solver = true) do state
        config_text = """
        [defaults]
        token = invalid-config-token
        endpoint = https://configured.example.invalid/sapi/
        """
        _with_temp_dwave_config(config_text) do _
            withenv("DWAVE_API_TOKEN" => nothing) do
                model = _build_two_spin_cloud_model()
                err = try
                    MOI.optimize!(model)
                    nothing
                catch e
                    e
                end

                Test.@test err isa ArgumentError
                msg = sprint(showerror, err)
                Test.@test occursin("The configured D-Wave cloud credentials or endpoint are not valid.", msg)
                Test.@test occursin("Invalid token", msg)
                Test.@test DWave.API_TOKEN[] === nothing
            end
        end

        counts_after = DWave.jl_object(state)
        Test.@test counts_after["client_tokens"] == Any["invalid-config-token"]
        Test.@test counts_after["client_endpoints"] == Any["https://configured.example.invalid/sapi/"]
        Test.@test counts_after["client_types"] == Any["qpu"]
        Test.@test counts_after["get_solver_calls"] == 1
        Test.@test counts_after["client_closed"] == 1
        Test.@test isempty(counts_after["sampler_tokens"])
    end

    _with_cloud_spies(; fail_solver = false) do state
        config_text = """
        [defaults]
        token = config-only-solve-token
        endpoint = https://configured.example.invalid/sapi/
        client = hybrid
        """
        _with_temp_dwave_config(config_text) do _
            withenv("DWAVE_API_TOKEN" => nothing) do
                model = _build_two_spin_cloud_model()
                MOI.optimize!(model)

                raw_solver = MOI.get(model, MOI.RawSolver())
                sampleset = QUBOTools.solution(raw_solver)
                Test.@test length(sampleset) == 1
                Test.@test DWave.API_TOKEN[] == "config-only-solve-token"
            end
        end

        counts_after = DWave.jl_object(state)
        Test.@test counts_after["client_tokens"] == Any["config-only-solve-token"]
        Test.@test counts_after["client_endpoints"] == Any["https://configured.example.invalid/sapi/"]
        Test.@test counts_after["client_types"] == Any["qpu"]
        Test.@test counts_after["get_solver_calls"] == 1
        Test.@test counts_after["client_closed"] == 1
        Test.@test counts_after["sampler_tokens"] == Any["config-only-solve-token"]
        Test.@test counts_after["sampler_endpoints"] == Any["https://configured.example.invalid/sapi/"]
        Test.@test counts_after["sampler_client_types"] == Any["qpu"]
        Test.@test counts_after["composite_calls"] == 1
        Test.@test counts_after["sample_ising_calls"] == 1
    end

    _with_cloud_spies(; fail_solver = false) do state
        withenv(
            "DWAVE_API_TOKEN" => "valid-cloud-token",
            "DWAVE_API_ENDPOINT" => "https://env.example.invalid/sapi/",
            "DWAVE_API_CLIENT" => "sw",
        ) do
            model = _build_two_spin_cloud_model()
            MOI.optimize!(model)

            raw_solver = MOI.get(model, MOI.RawSolver())
            sampleset = QUBOTools.solution(raw_solver)
            Test.@test length(sampleset) == 1
            Test.@test DWave.API_TOKEN[] == "valid-cloud-token"

            counts_after = DWave.jl_object(state)
            Test.@test counts_after["client_tokens"] == Any["valid-cloud-token"]
            Test.@test counts_after["client_endpoints"] == Any["https://env.example.invalid/sapi/"]
            Test.@test counts_after["client_types"] == Any["qpu"]
            Test.@test counts_after["get_solver_calls"] == 1
            Test.@test counts_after["client_closed"] == 1
            Test.@test counts_after["sampler_tokens"] == Any["valid-cloud-token"]
            Test.@test counts_after["sampler_endpoints"] == Any["https://env.example.invalid/sapi/"]
            Test.@test counts_after["sampler_client_types"] == Any["qpu"]
            Test.@test counts_after["composite_calls"] == 1
            Test.@test counts_after["sample_ising_calls"] == 1
        end
    end

    # Real Ocean Client.from_config dispatch with mocked QPU solver session and sw/hybrid client config
    prev_token = DWave.API_TOKEN[]
    try
        for configured_client in ("sw", "hybrid")
            for source in (:file, :env)
                cfg = """
                [defaults]
                token = ocean-dispatch-token
                endpoint = https://configured.example.invalid/sapi/
                """
                if source === :file
                    cfg *= "client = $(configured_client)\n"
                end
                _with_in_memory_dwave_config(cfg) do _
                    withenv(
                        "DWAVE_API_TOKEN" => nothing,
                        "DWAVE_PROFILE" => nothing,
                        "DWAVE_API_CLIENT" => (source === :env ? configured_client : nothing),
                    ) do
                        patch_guard = DWave.PythonCall.pyexec(
                            @NamedTuple{stop::DWave.PythonCall.Py},
                            """
def install_real_ocean_qpu_mock():
    from unittest.mock import patch
    from dwave.cloud import Client
    from dwave.cloud.api.models import SolverConfiguration
    solver = SolverConfiguration.model_validate({
        "identity": {"name": "mock-qpu"},
        "status": "online",
        "avg_load": 0,
        "properties": {
            "category": "qpu",
            "supported_problem_types": ["ising", "qubo"],
            "qubits": [0, 1],
            "couplers": [[0, 1]],
            "num_qubits": 2,
            "parameters": {"num_reads": []},
        },
    })
    class SolversSession:
        def list_solvers(self, **kwargs):
            return [solver]
        def get_solver(self, **kwargs):
            return solver
    p = patch.object(Client, "solvers_session", property(lambda self: SolversSession()))
    p.start()
    return p.stop
stop = install_real_ocean_qpu_mock()
""",
                            @__MODULE__,
                            (),
                        )
                        try
                            DWave.API_TOKEN[] = nothing
                            Test.@test DWave.__auth__(; verbose = false)
                            Test.@test DWave.API_TOKEN[] == "ocean-dispatch-token"
                            sampler = DWave._default_cloud_sampler()
                            try
                                Test.@test DWave.PythonCall.pyconvert(
                                    String,
                                    sampler.child.solver.name,
                                ) == "mock-qpu"
                            finally
                                sampler.child.close()
                            end
                        finally
                            patch_guard.stop()
                        end
                    end
                end
            end
        end
    finally
        DWave.API_TOKEN[] = prev_token
    end
end
