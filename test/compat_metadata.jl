import DWave
import QUBODrivers
import TOML
import Test

const PACKAGE_ROOT = normpath(joinpath(@__DIR__, ".."))
const CompatMOI = QUBODrivers.MOI

function _compat_entries(value::String)
    return strip.(split(value, ','))
end

Test.@testset "Compatibility metadata matches supported JuliaQUBO stack" begin
    compat = TOML.parsefile(joinpath(PACKAGE_ROOT, "Project.toml"))["compat"]

    Test.@test compat["julia"] == "1.10"
    Test.@test _compat_entries(compat["QUBODrivers"]) == ["0.6"]
    Test.@test _compat_entries(compat["QUBOTools"]) == ["0.13", "0.14", "0.15", "0.16"]
end

Test.@testset "QUBODrivers 0.6 capability traits are declared" begin
    Test.@test !QUBODrivers.supports_seed(DWave.Optimizer)
    Test.@test QUBODrivers.supports_seed(DWave.Neal.Optimizer)
    Test.@test QUBODrivers.supports_seed(DWave.Greedy.Optimizer)
    Test.@test QUBODrivers.supports_seed(DWave.Random.Optimizer)
    Test.@test QUBODrivers.supports_seed(DWave.Tabu.Optimizer)

    Test.@test QUBODrivers.honors_final_reads(DWave.Optimizer)
    Test.@test QUBODrivers.honors_final_reads(DWave.Neal.Optimizer)
    Test.@test QUBODrivers.honors_final_reads(DWave.Greedy.Optimizer)
    Test.@test QUBODrivers.honors_final_reads(DWave.Random.Optimizer)
    Test.@test QUBODrivers.honors_final_reads(DWave.Tabu.Optimizer)

    Test.@test !QUBODrivers.enforces_time_limit(DWave.Optimizer)
    Test.@test !QUBODrivers.enforces_time_limit(DWave.Neal.Optimizer)
    Test.@test !QUBODrivers.enforces_time_limit(DWave.Greedy.Optimizer)
    Test.@test QUBODrivers.enforces_time_limit(DWave.Random.Optimizer)
    Test.@test !QUBODrivers.enforces_time_limit(DWave.Tabu.Optimizer)
end

Test.@testset "RandomSeed aliases existing seed attributes" begin
    model = CompatMOI.instantiate(DWave.Neal.Optimizer; with_bridge_type = Float64)

    Test.@test CompatMOI.supports(model, QUBODrivers.RandomSeed())
    Test.@test isnothing(CompatMOI.get(model, QUBODrivers.RandomSeed()))

    CompatMOI.set(model, QUBODrivers.RandomSeed(), 12_345)

    Test.@test CompatMOI.get(model, QUBODrivers.RandomSeed()) == 12_345
    Test.@test CompatMOI.get(model, CompatMOI.RawOptimizerAttribute("seed")) == 12_345
end

Test.@testset "CI covers Julia floor and latest stable" begin
    workflow = read(joinpath(PACKAGE_ROOT, ".github", "workflows", "ci.yml"), String)

    Test.@test occursin(r"(?m)^\s*-\s*'1\.10'\s*$", workflow)
    Test.@test occursin(r"(?m)^\s*-\s*'1'\s*$", workflow)
    Test.@test !occursin("1.12", workflow)
end

Test.@testset "Dependency maintenance automation follows policy" begin
    dependabot = read(joinpath(PACKAGE_ROOT, ".github", "dependabot.yml"), String)
    workflows = readdir(joinpath(PACKAGE_ROOT, ".github", "workflows"))

    Test.@test occursin(r"(?m)^\s*-\s*package-ecosystem:\s*\"julia\"\s*$", dependabot)
    Test.@test occursin(r"(?m)^\s*directory:\s*\"/\"\s*$", dependabot)
    Test.@test occursin("root-julia-dependencies", dependabot)
    Test.@test occursin(r"(?m)^\s*-\s*package-ecosystem:\s*\"github-actions\"\s*$", dependabot)
    Test.@test occursin(r"(?m)^\s*interval:\s*\"monthly\"\s*$", dependabot)
    Test.@test !occursin(r"(?m)^\s*directory:\s*\"/docs\"\s*$", dependabot)
    Test.@test !occursin(r"(?m)^\s*directory:\s*\"/test\"\s*$", dependabot)
    Test.@test !any(name -> occursin("compathelper", lowercase(name)), workflows)
end

Test.@testset "Python dependencies match audited stable stack" begin
    condapkg = TOML.parsefile(joinpath(PACKAGE_ROOT, "CondaPkg.toml"))
    pip_deps = condapkg["pip"]["deps"]

    Test.@test condapkg["deps"]["python"] == ">=3.11,<=3.13"
    Test.@test !haskey(pip_deps, "numpy")
    Test.@test pip_deps["dwave-ocean-sdk"] == "==9.3.0"
    Test.@test pip_deps["dwave_networkx"] == "==0.8.18"
end

Test.@testset "Python dependency policy allows shared benchmark env" begin
    condapkg = TOML.parsefile(joinpath(PACKAGE_ROOT, "CondaPkg.toml"))
    pip_deps = condapkg["pip"]["deps"]

    # Mirrors the registered Python-backed benchmark tier from issue #55.
    cimoptimizer_conda_deps = Dict(
        "python" => ">=3.10,<3.12",
        "pytorch-cpu" => ">=2.0.1",
    )
    cimoptimizer_pip_deps = Dict("cim-optimizer" => "==1.0.4")
    qiskitopt_pip_deps = Dict(
        "qiskit" => "~=2.3.0",
        "qiskit-aer" => "~=0.17.0",
        "qiskit-ibm-runtime" => "~=0.46.0",
        "qiskit-optimization" => "~=0.7.0",
        "scipy" => "~=1.15.0",
    )
    pysa_pip_deps = Dict(
        "numpy" => ">=1.20.0",
        "pysa" => "@ git+https://github.com/nasa/pysa@v0.1.0",
    )

    overlapping_pip_packages = Set(
        intersect(
            collect(keys(pip_deps)),
            union(
                collect(keys(cimoptimizer_pip_deps)),
                collect(keys(qiskitopt_pip_deps)),
                collect(keys(pysa_pip_deps)),
            ),
        ),
    )

    Test.@test haskey(cimoptimizer_conda_deps, "pytorch-cpu")
    Test.@test !haskey(pip_deps, "numpy")
    Test.@test overlapping_pip_packages == Set{String}()

    merged_pip_deps = merge(cimoptimizer_pip_deps, qiskitopt_pip_deps, pysa_pip_deps, pip_deps)

    Test.@test merged_pip_deps["numpy"] == ">=1.20.0"
    Test.@test merged_pip_deps["dwave-ocean-sdk"] == "==9.3.0"
    Test.@test merged_pip_deps["dwave_networkx"] == "==0.8.18"
    Test.@test merged_pip_deps["cim-optimizer"] == "==1.0.4"
end
