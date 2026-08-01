import Test
import TOML

const CITATION_PACKAGE_ROOT = normpath(joinpath(@__DIR__, ".."))

include(joinpath(CITATION_PACKAGE_ROOT, "scripts", "verify_zenodo_release.jl"))

function _zenodo_record(; concept = "7812066", version = "v0.7.6")
    return Dict(
        "conceptrecid" => concept,
        "doi" => "10.5281/zenodo.21078136",
        "metadata" => Dict("version" => version),
    )
end

Test.@testset "Citation metadata is consistent" begin
    read_text(path) = replace(read(path, String), "\r\n" => "\n")

    project = TOML.parsefile(joinpath(CITATION_PACKAGE_ROOT, "Project.toml"))
    citation = read_text(joinpath(CITATION_PACKAGE_ROOT, "CITATION.cff"))
    readme = read_text(joinpath(CITATION_PACKAGE_ROOT, "README.md"))
    checklist = read_text(joinpath(CITATION_PACKAGE_ROOT, ".github", "RELEASE_CHECKLIST.md"))
    workflow = read_text(joinpath(CITATION_PACKAGE_ROOT, ".github", "workflows", "zenodo.yml"))

    concept_doi = "10.5281/zenodo.7812066"
    version_doi = "10.5281/zenodo.21078136"
    article_doi = "10.1080/10556788.2026.2702926"
    version = string(project["version"])

    Test.@test occursin("doi: \"$concept_doi\"", citation)
    Test.@test occursin("version: \"$version\"", citation)
    Test.@test occursin(article_doi, citation)

    Test.@test occursin("badge/DOI/$concept_doi.svg", readme)
    Test.@test occursin("doi.org/$concept_doi", readme)
    Test.@test occursin("doi.org/$version_doi", readme)
    Test.@test occursin("doi.org/$article_doi", readme)

    Test.@test occursin(concept_doi, checklist)
    Test.@test occursin("Can manage", checklist)
    Test.@test occursin("Zenodo release verification", checklist)
    Test.@test occursin(r"(?m)^\s*release:\s*$", workflow)
    Test.@test occursin(r"(?m)^\s*-\s*published\s*$", workflow)
    Test.@test occursin("scripts/verify_zenodo_release.jl", workflow)
end

Test.@testset "Zenodo release comparison detects archive drift" begin
    current = ZenodoReleaseVerification.check_record(_zenodo_record(), "v0.7.6")
    stale = ZenodoReleaseVerification.check_record(
        _zenodo_record(; version = "v0.7.5"),
        "v0.7.6",
    )
    wrong_concept = ZenodoReleaseVerification.check_record(
        _zenodo_record(; concept = "9999999"),
        "v0.7.6",
    )

    Test.@test current.ok
    Test.@test current.doi == "10.5281/zenodo.21078136"
    Test.@test !stale.ok
    Test.@test occursin("v0.7.5", stale.message)
    Test.@test !wrong_concept.ok
    Test.@test occursin("9999999", wrong_concept.message)

    responses = Any[_zenodo_record(; version = "v0.7.5"), _zenodo_record()]
    result = ZenodoReleaseVerification.verify_release(
        "v0.7.6";
        attempts = 2,
        delay_seconds = 0,
        record_fetcher = _ -> popfirst!(responses),
    )

    Test.@test result.ok
    Test.@test isempty(responses)
    Test.@test_throws ErrorException ZenodoReleaseVerification.verify_release(
        "v0.7.6";
        attempts = 1,
        delay_seconds = 0,
        record_fetcher = _ -> _zenodo_record(; version = "v0.7.5"),
    )

    transport_error = try
        ZenodoReleaseVerification.verify_release(
            "v0.7.6";
            attempts = 1,
            delay_seconds = 0,
            record_fetcher = _ -> error("network unavailable"),
        )
        nothing
    catch error
        sprint(showerror, error)
    end

    Test.@test occursin("network unavailable", transport_error)
end
