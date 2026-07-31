module ZenodoReleaseVerification

import Downloads
import JSON

const CONCEPT_RECORD_ID = "7812066"
const DEFAULT_API_URL = "https://zenodo.org/api/records/$CONCEPT_RECORD_ID/versions/latest"

normalize_version(value) = replace(strip(string(value)), r"^[vV]" => "")

function check_record(record::AbstractDict, expected_tag::AbstractString)
    concept_record_id = string(get(record, "conceptrecid", ""))
    if concept_record_id != CONCEPT_RECORD_ID
        return (
            ok = false,
            message = "Zenodo returned concept record '$concept_record_id', expected '$CONCEPT_RECORD_ID'.",
            doi = nothing,
        )
    end

    metadata = get(record, "metadata", nothing)
    if !(metadata isa AbstractDict) || !haskey(metadata, "version")
        return (
            ok = false,
            message = "Zenodo's latest record does not declare metadata.version.",
            doi = nothing,
        )
    end

    archived_version = string(metadata["version"])
    if normalize_version(archived_version) != normalize_version(expected_tag)
        return (
            ok = false,
            message = "Zenodo's latest version is '$archived_version', expected '$expected_tag'.",
            doi = get(record, "doi", nothing),
        )
    end

    return (
        ok = true,
        message = "Zenodo record $(get(record, "doi", "without a DOI")) matches release '$expected_tag'.",
        doi = get(record, "doi", nothing),
    )
end

function fetch_record(url::AbstractString)
    output = IOBuffer()
    response = Downloads.request(url; output, timeout = 30, throw = false)
    hasproperty(response, :status) || error(sprint(showerror, response))
    response.status == 200 || error("Zenodo returned HTTP $(response.status).")

    return JSON.parse(String(take!(output)))
end

function verify_release(
    expected_tag::AbstractString;
    api_url::AbstractString = get(ENV, "ZENODO_API_URL", DEFAULT_API_URL),
    attempts::Int = parse(Int, get(ENV, "ZENODO_VERIFY_ATTEMPTS", "12")),
    delay_seconds::Real = parse(Float64, get(ENV, "ZENODO_VERIFY_DELAY_SECONDS", "30")),
    record_fetcher = fetch_record,
)
    isempty(strip(expected_tag)) && throw(ArgumentError("The expected release tag is empty."))
    attempts > 0 || throw(ArgumentError("ZENODO_VERIFY_ATTEMPTS must be positive."))
    delay_seconds >= 0 || throw(ArgumentError("ZENODO_VERIFY_DELAY_SECONDS must be nonnegative."))

    last_failure = "No verification attempt ran."

    for attempt in 1:attempts
        try
            result = check_record(record_fetcher(api_url), expected_tag)
            result.ok && return result
            last_failure = result.message
        catch error
            last_failure = sprint(showerror, error)
        end

        if attempt < attempts
            println(
                stderr,
                "Zenodo verification attempt $attempt/$attempts failed: $last_failure Retrying in $(delay_seconds) seconds.",
            )
            sleep(delay_seconds)
        end
    end

    error("Zenodo release verification failed after $attempts attempts: $last_failure")
end

function main(args)
    length(args) == 1 || throw(ArgumentError("Usage: verify_zenodo_release.jl <release-tag>"))
    result = verify_release(only(args))
    println(result.message)

    return nothing
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    try
        ZenodoReleaseVerification.main(ARGS)
    catch error
        println(stderr, "ERROR: ", sprint(showerror, error))
        exit(1)
    end
end
