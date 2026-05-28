import TOML
import Test

const PACKAGE_ROOT = normpath(joinpath(@__DIR__, ".."))

function _compat_entries(value::String)
    return strip.(split(value, ','))
end

Test.@testset "Compatibility metadata matches supported JuliaQUBO stack" begin
    compat = TOML.parsefile(joinpath(PACKAGE_ROOT, "Project.toml"))["compat"]

    Test.@test compat["julia"] == "1.10"
    Test.@test "0.3.3" in _compat_entries(compat["QUBODrivers"])
    Test.@test "0.4" in _compat_entries(compat["QUBODrivers"])
    Test.@test "0.5" in _compat_entries(compat["QUBODrivers"])
    Test.@test "0.10" in _compat_entries(compat["QUBOTools"])
    Test.@test "0.11" in _compat_entries(compat["QUBOTools"])
    Test.@test "0.12" in _compat_entries(compat["QUBOTools"])
end

Test.@testset "CI covers Julia floor and latest stable" begin
    workflow = read(joinpath(PACKAGE_ROOT, ".github", "workflows", "ci.yml"), String)

    Test.@test occursin(r"(?m)^\s*-\s*'1\.10'\s*$", workflow)
    Test.@test occursin(r"(?m)^\s*-\s*'1'\s*$", workflow)
    Test.@test !occursin("1.12", workflow)
end

Test.@testset "Python dependencies match audited stable stack" begin
    condapkg = TOML.parsefile(joinpath(PACKAGE_ROOT, "CondaPkg.toml"))
    pip_deps = condapkg["pip"]["deps"]

    Test.@test condapkg["deps"]["python"] == ">=3.11,<=3.13"
    Test.@test pip_deps["numpy"] == "==2.4.6"
    Test.@test pip_deps["dwave-ocean-sdk"] == "==9.3.0"
    Test.@test pip_deps["dwave_networkx"] == "==0.8.18"
end
