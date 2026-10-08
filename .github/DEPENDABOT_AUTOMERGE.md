# Dependabot automatic merges

All same-repository Dependabot-authored PRs to `main`, including major and grouped
updates, are eligible when their current head passes the checks below and every
additional reported check has finished successfully (neutral/skipped optional
checks are allowed). Required checks cannot be skipped. Human PRs, drafts,
forks, changed heads, conflicts, and behind branches are left to maintainers.
Identity follows the PR author: maintainer commits on a Dependabot branch are
eligible only after that head passes the same gates.

The workflow checks out trusted `main`, never PR code or artifacts, and uses the
built-in Actions token. It re-reads identity and clean mergeability, verifies
enforced required checks and that the verified head contains current `main`,
and squash-merges with `--match-head-commit`, without
admin bypass or a deferred merge request. Existing review/conversation gates
still apply. Errors are isolated per PR and reported after both merge and
publication passes; one failure cannot starve the other PRs.

## Required successful checks

- `Julia 1.10 - ubuntu-latest - x64 - pull_request`
- `Julia 1.10 - windows-latest - x64 - pull_request`
- `Julia 1 - ubuntu-latest - x64 - pull_request`
- `Julia 1 - windows-latest - x64 - pull_request`
- `Dependabot policy`

## Branch protection and activation

The existing required CI checks remain unchanged.
The workflow verifies public branch-protection metadata and SHA ancestry.
The public branch summary omits the administration-only `strict` field. Strict
up-to-date checking remains an activation prerequisite, and GitHub enforces
protection atomically when merging. The script requires protected/enabled
enforcement and refuses any head that does not contain current `main`.
These required contexts must remain enforced:

- `Julia 1.10 - ubuntu-latest - x64 - pull_request`
- `Julia 1.10 - windows-latest - x64 - pull_request`
- `Julia 1 - ubuntu-latest - x64 - pull_request`
- `Julia 1 - windows-latest - x64 - pull_request`

When `main` advances, update
existing Dependabot branches with "Update branch" or `@dependabot rebase` and
wait for new CI. The policy runs after configured workflow completions, every
15 minutes, and through manual dispatch. Repository native auto-merge settings
only control manually queued requests; this workflow merges immediately after
verifying all gates.

## Publication and live verification

`GITHUB_TOKEN` merges suppress ordinary push/PR-close workflows. This repository has no documentation publisher to dispatch after token merges.
Main push CI is also suppressed. The up-to-date PR CI remains required.
Tag/release publishing retains its existing triggers;
this policy does not invoke release workflows.

After activation, verify the first real token merge at its merge SHA. GitHub documents Contents write access
for the [PR merge API](https://docs.github.com/en/rest/pulls/pulls#merge-a-pull-request).
Actions-file updates use the same path; token permissions and completion events
still need live verification. If GitHub rejects such a merge, a maintainer merges
that PR manually; its visible error does not block other PRs. No PAT or new secret
is required. Do not force-push publication history or hide failures to obtain green CI.
