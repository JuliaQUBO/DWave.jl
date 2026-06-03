import Test

const AUTH_PACKAGE_ROOT = normpath(joinpath(@__DIR__, ".."))

function _empty_dwave_token_checks_pass()
    code = """
    using DWave
    @assert DWave.API_TOKEN[] === nothing

    DWave.API_TOKEN[] = "stale-token"
    ENV["DWAVE_API_TOKEN"] = "   "
    @assert !DWave.__auth__(; verbose = false)
    @assert DWave.API_TOKEN[] === nothing
    """
    julia = joinpath(Sys.BINDIR, Base.julia_exename())
    auth_depot = joinpath(tempdir(), "dwave-auth-test-depot")
    depot_separator = Sys.iswindows() ? ";" : ":"

    mkpath(auth_depot)
    depot_path = string(auth_depot, depot_separator, join(DEPOT_PATH, depot_separator))

    return withenv("DWAVE_API_TOKEN" => "", "JULIA_DEPOT_PATH" => depot_path) do
        success(`$julia --startup-file=no --compiled-modules=no --project=$(AUTH_PACKAGE_ROOT) -e $code`)
    end
end

Test.@testset "Empty D-Wave API token is ignored during initialization" begin
    Test.@test _empty_dwave_token_checks_pass()
end
