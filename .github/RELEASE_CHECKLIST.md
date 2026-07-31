# Release checklist

Use this checklist for every tagged release. The Zenodo concept DOI
`10.5281/zenodo.7812066` is the permanent DWave.jl software identifier; do not
create a new concept record for a new version.

## Before tagging

- [ ] Confirm that at least two active JuliaQUBO maintainers have **Can manage**
      access to the existing Zenodo deposit. Record the stewards and review
      date in the ecosystem tracker, JuliaQUBO/QUBO.jl#66.
- [ ] Update the version in `Project.toml` and the version and release date in
      `CITATION.cff`.
- [ ] Keep the concept DOI in `CITATION.cff` and the README badge unchanged.
- [ ] Validate the citation metadata with
      `cffconvert --validate --infile CITATION.cff`.
- [ ] Run the package test suite with
      `julia --project -e 'using Pkg; Pkg.test()'`.

## After publishing the GitHub release

- [ ] Confirm the **Zenodo release verification** workflow passes. It retries
      for six minutes and then fails if the latest record under concept DOI
      `10.5281/zenodo.7812066` does not match the GitHub release tag.
- [ ] If Zenodo publication outlasts that bounded retry window, wait for the
      archive to appear and rerun the workflow manually with the release tag.
      A persistent mismatch means the release integration needs repair; it is
      not a reason to mint a new concept DOI.
- [ ] Confirm the record's version, MIT license, repository URL, Julia package
      UUID (`4d534982-bf11-4157-9e48-fe3a62208a50`), creators, and archive
      contents.
- [ ] Record the new version DOI in the GitHub release notes and update the
      README's exact-version example.
- [ ] Download the Zenodo archive and confirm its `Project.toml` version matches
      the GitHub tag.
