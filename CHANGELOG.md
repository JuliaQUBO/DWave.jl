# Release notes

## 0.8.0 — 2026-10-11

This breaking minor release corrects Neal read accounting, interruption status,
and temperature semantics. Neal callers that use evaluation counts or termination
statuses should review the accounting changes below, including through MOI/JuMP.
Directly installed QUBOTools models with nonunit scale can change behavior with
existing explicit schedules; negative scale changes the optimization direction
regardless of schedule.

### Read accounting and interruption

- Neal metadata distinguishes requested reads (`reads.number_of_reads`), the
  effective allocation (`reads.final_number_of_reads`), and completed reads
  (`reads.completed_number_of_reads`, summed returned multiplicities).
  `optimizer.evaluations` now counts completed returned reads, with
  `optimizer.evaluation_unit = "returned_reads"`. Consumers needing the allocation
  should use the retained allocation fields. These are not proposal or spin-flip
  counts.
- An actual truthy interrupt callback reports `MOI.INTERRUPTED`, including when
  it fires after the final allotted read. A short return alone reports
  `MOI.OTHER_ERROR`, not an inferred interruption or time limit. These cases
  previously reported `MOI.LOCALLY_SOLVED`. Native callback
  errors report `MOI.OTHER_ERROR` while preserving completed results; propagated
  backend exceptions clear stale results and report a sampling error.
- Valid returned incumbents remain available through the public MOI result,
  primal, and objective accessors after interruption or native callback errors.
  Optimizer reuse starts with fresh termination state. Other samplers retain
  their existing accounting and status behavior.

### Temperature semantics and migration

- Neal inverse temperatures (`beta_range` and custom `beta_schedule`) are in
  inverse units of the original objective energy. For
  `f(x) = scale * (polynomial(x) + offset)`, Neal anneals `f` when minimizing and
  `-f` when maximizing. Scale is absorbed into the original sparse coefficients
  before domain conversion; explicit beta values are forwarded unchanged.
- Negative scale now reverses the native energy landscape correctly. Zero scale
  gives zero native biases and zero returned objectives without division by
  scale. Constants do not affect transitions and are restored in returned
  objective values. Equivalent direct QUBOTools and MOI loading use the same
  scaling/conversion contract, including automatic schedules.
- **Compatibility:** directly installed nonunit-scale QUBOTools models can follow
  different trajectories with old explicit schedules. Ordinary MOI/JuMP loading
  already absorbs scale into coefficients; its unit-scale temperature semantics
  remain unchanged. Unit-scale direct models also retain their semantics.
- **Positive-scale migration:** for scale `1.5`, the old explicit
  `beta_range = (0.3, 3.0)` had effective original-energy temperatures corresponding
  to `(0.2, 2.0)`. Use `beta_range = (0.2, 2.0)` under the corrected contract to
  retain that effective temperature. Likewise divide each old custom beta by
  `1.5`. The old incorrect negative-scale optimization direction is not preserved.
- With `beta_range = nothing`, upstream Neal derives linear/geometric ranges from
  the scaled native biases. In exact arithmetic its current rule divides the
  automatic endpoints by `abs(scale)` for a nonzero uniform scale; floating-point
  bias ties can affect endpoints. Negative scale still reverses the landscape.
  All-zero/constant objectives use upstream's arbitrary `[0.1, 1.0]` range and
  zero-bias warning. Custom schedules use the supplied values.

### Other changes since 0.7.6

- Added software citation metadata and a bounded Zenodo archive-verification
  workflow, retaining the MIT license and concept DOI
  [10.5281/zenodo.7812066](https://doi.org/10.5281/zenodo.7812066).
- Improved compatibility metadata tests and dependency-update CI policy; updated
  checkout automation and disabled unused coverage collection. This release
  preparation does not change dependency bounds.

The correctness and parity checks do not establish decomposition performance or
certify a downstream benchmark campaign. Downstream adoption of 0.8.0 requires a
fresh validation gate for that dependency and runtime.

Implementation: [#79](https://github.com/JuliaQUBO/DWave.jl/pull/79),
[#80](https://github.com/JuliaQUBO/DWave.jl/pull/80).
