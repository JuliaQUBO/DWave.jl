# DWave.jl
[![CI](https://github.com/JuliaQUBO/DWave.jl/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/JuliaQUBO/DWave.jl/actions/workflows/ci.yml?query=branch%3Amain)
[![QUBODRIVERS](https://img.shields.io/badge/Powered%20by-QUBODrivers.jl-%20%234063d8)](https://github.com/JuliaQUBO/QUBODrivers.jl)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7812066.svg)](https://doi.org/10.5281/zenodo.7812066)

D-Wave Quantum Annealing Interface for JuMP

## Installation
```julia
julia> import Pkg

julia> Pkg.add("DWave")
```

## Migration from DWaveNeal.jl
`DWaveNeal.jl` only covered the simulated annealing wrapper that now lives at
`DWave.Neal.Optimizer` inside `DWave.jl`. The rest of `DWave.jl` provides
additional samplers and is not part of the old `DWaveNeal.jl` API surface.

For code that used `DWaveNeal.Optimizer`, install `DWave.jl` and switch to
`DWave.Neal.Optimizer`:

```julia
julia> import Pkg

julia> Pkg.add("DWave")

julia> using DWave
```

If you still have an environment that depends on `DWaveNeal`, `Pkg.add("DWaveNeal")`
continues to work, but `DWaveNeal.Optimizer` now aliases
`DWave.Neal.Optimizer` and emits a deprecation warning on load.

## Basic Usage
```julia
using JuMP
using QUBO
using DWave

model = Model(DWave.Neal.Optimizer)

h = [-1, -1, -1]
J = [0 2 2; 0 0 2; 0 0 0]

@variable(model, s[1:3], Spin)

@objective(model, Min, h's + s'J * s)

optimize!(model)

for i = 1:result_count(model)
    si = value.(s; result=i)
    yi = objective_value(model; result=i)

    println("H($si) = $yi")
end
```

## Sampler Overview
`DWave.jl` provides two families of optimizers:

- `DWave.Optimizer` connects to D-Wave cloud samplers and requires access to a
  Leap account.
- `DWave.Neal.Optimizer`, `DWave.Greedy.Optimizer`,
  `DWave.Random.Optimizer`, and `DWave.Tabu.Optimizer` wrap the classical
  samplers shipped in `dwave-samplers` and run locally through the
  PythonCall/CondaPkg environment managed by the package.

Switching between them only requires changing the optimizer passed to
`Model(...)`.

## Classical Samplers
The classical samplers expose the same QUBO/Ising modeling interface as the QPU
wrapper, but they do not require `DWAVE_API_TOKEN`.

- `DWave.Neal.Optimizer`
- `DWave.Greedy.Optimizer`
- `DWave.Random.Optimizer`
- `DWave.Tabu.Optimizer`

Example:

```julia
using JuMP
using QUBO
using DWave

model = Model(DWave.Tabu.Optimizer)

set_attribute(model, "num_reads", 32)
set_attribute(model, "timeout", 100)
set_attribute(model, "initial_states", [
     1  1 -1 -1
    -1 -1  1  1
])

h = [-1, -1, 0, 0]
J = [0 0 1 0; 0 0 0 1; 0 0 0 0; 0 0 0 0]

@variable(model, s[1:4], Spin)
@objective(model, Min, h' * s + s' * J * s)

optimize!(model)
```

Sampler-specific options are forwarded as raw optimizer attributes via
`set_attribute(model, name, value)`. For `initial_states`, the Greedy and Tabu
wrappers accept either a single Julia vector or a matrix whose rows are initial
states.

| Optimizer | Description | Raw optimizer attributes |
| --- | --- | --- |
| `DWave.Neal.Optimizer` | Simulated annealing baseline. | `num_reads`, `num_sweeps`, `num_sweeps_per_beta`, `beta_range`, `beta_schedule`, `beta_schedule_type`, `seed`, `initial_states_generator`, `interrupt_function` |
| `DWave.Greedy.Optimizer` | Steepest-descent local search. | `num_reads`, `initial_states`, `initial_states_generator`, `seed`, `large_sparse_opt` |
| `DWave.Random.Optimizer` | Random-state sampling. | `num_reads`, `time_limit`, `max_num_samples`, `seed` |
| `DWave.Tabu.Optimizer` | Tabu-search local search. | `initial_states`, `initial_states_generator`, `num_reads`, `seed`, `tenure`, `timeout`, `num_restarts`, `energy_threshold`, `coefficient_z_first`, `coefficient_z_restart`, `lower_bound_z` |

The upstream `planar` and `tree` samplers are not wrapped yet. `PlanarGraphSolver`
only applies to planar Ising models without linear biases, and the tree
decomposition samplers expose exact-solver and marginal APIs that do not fit the
current `QUBODrivers` interface cleanly.

### Neal temperature and objective energy units

Neal's inverse temperatures (`beta_range` or a custom `beta_schedule`) are in
inverse units of the original objective energy. If the model represents
`f(x) = scale * (polynomial(x) + offset)`, Neal anneals `f` for minimization
and `-f` for maximization. Model scale is absorbed into the sparse native
coefficients before converting Bool to Spin, matching MOI loading arithmetic;
the explicit beta values are forwarded unchanged. A negative
scale reverses the native coefficients and hence the optimization direction.
Sense conversion is applied once. Additive constants are excluded from the
native BQM because they do not affect transitions, and are restored in returned
objective values. Zero scale produces zero native biases and zero objective
values without division by scale.

Equivalent polynomials loaded as QUBOTools models or copied through MOI thus
have equivalent native energy differences and temperature schedules, up to
floating-point rounding during conversion.

With `beta_range = nothing`, upstream Neal chooses its range from the scaled
native biases for both `"linear"` and `"geometric"` schedules. In the current
upstream rule in exact arithmetic, a nonzero uniform scale changes the automatic beta endpoints by
`1 / abs(scale)`: absolute incident biases determine the endpoints. Thus the
beta times energy-difference magnitudes compensate for uniform scaling; negative
scale still reverses the energy landscape. Floating-point summation can affect
the exact bias ties used by upstream and hence the chosen endpoints. For an
all-zero or constant objective,
upstream uses the arbitrary range `[0.1, 1.0]` and warns about zero biases.
Custom schedules use `beta_schedule_type = "custom"` and the supplied values.

This corrects earlier behavior that sampled unscaled coefficients and applied
scale only when formatting results. Directly installed QUBOTools models with
nonunit scale can therefore produce different trajectories with existing explicit
schedules. Ordinary MOI/JuMP loading already absorbs scale into its coefficients
and retains unit-scale temperature semantics. For a positive scale of
`1.5`, an old `beta_range = (0.3, 3.0)` had the effective original-energy range
`(0.2, 2.0)`; use `(0.2, 2.0)` to retain that effective temperature after this
change. Likewise divide each old custom beta by `1.5`. No preservation of the
old incorrect negative-scale optimization direction is promised. Unit-scale
models retain their temperature semantics.

### Neal read accounting and interruption

Neal result metadata distinguishes allocation from returned work:

| Metadata field | Meaning |
| --- | --- |
| `reads.number_of_reads` | Requested `num_reads`. |
| `reads.final_number_of_reads` | Effective allocation forwarded to Neal, including a `QUBODrivers.FinalNumberOfReads()` override. |
| `reads.completed_number_of_reads` | Sum of returned `num_occurrences`, including repeated samples. |
| `optimizer.evaluations` | Completed returned reads; previously this reported the effective allocation. |
| `optimizer.evaluation_unit` | `"returned_reads"`; these are not native proposal or spin-flip counts. |

For example, requesting 11 reads with a final allocation of 7 and stopping after
one read records `11`, `7`, and `1` respectively, with one evaluation. Two
aggregated rows with multiplicities 4 and 3 record seven completed reads.

The `interrupt_function` is called only by the upstream sampler, with no arguments
after each completed read, including the final one. Solve-local tracking forwards
its arguments and uses the native Python truth conversion of its return value.
It adds no callback calls or work inside annealing sweeps. `termination.interrupt_requested` records
whether an actual callback returned a truthy value, independently of the number
of returned solutions. Such a solve reports `MOI.INTERRUPTED`, raw status
`"interrupt_callback"`, and `termination.reason = "interrupt_callback"`. This
also applies when the callback returns true after the final allotted read: the
allocation is complete, but the callback requested termination. An arbitrary
callback does not imply a time limit.

Normal full completion reports `MOI.LOCALLY_SOLVED` and raw status
`"locally_solved"`. A short or empty return without callback interruption reports
`MOI.OTHER_ERROR` and `"incomplete_return"`; it does not establish interruption.
The native Neal bridge treats callback exceptions as a stop and returns the
reads completed so far. Tracking preserves that behavior and records
`termination.interrupt_callback_error = true`, `MOI.OTHER_ERROR`, and raw status
`"interrupt_callback_error"`. Exceptions propagated by a backend are rethrown;
the previous solve's results are cleared and public status becomes
`MOI.OTHER_ERROR` with `"sampling_error"`.

Valid returned incumbents remain accessible through `MOI.ResultCount()`,
`MOI.PrimalStatus()` (`FEASIBLE_POINT` for valid result indices),
`MOI.VariablePrimal()`, and `MOI.ObjectiveValue()`, including after interruption
or a native callback error. A fresh or emptied optimizer reports
`MOI.OPTIMIZE_NOT_CALLED`; each new solve starts with fresh termination state.
These accounting and status corrections apply to Neal. Other classical and
QPU samplers retain their existing metadata and status behavior.

## API Token
To use D-Wave's QPU it is necessary to obtain an API Token from [Leap](https://cloud.dwavesys.com/leap/).

## Topology and Embedding Plots
`DWave.draw_topology` and `DWave.draw_embedding` wrap D-Wave NetworkX's
Pegasus and Zephyr plotting helpers and return a Matplotlib figure by default.
Both helpers accept `DWave.WorkingGraph` values built from sampler metadata, so
plots use the full calibrated working graph rather than only the embedded
qubits.

```julia
using DWave
using QUBOTools

metadata = QUBOTools.metadata(sampleset)

DWave.draw_topology(DWave.WorkingGraph(metadata))
DWave.draw_embedding(metadata)
```

Pass `ax = existing_axis` to draw into an existing Matplotlib axis; all other
keyword arguments are forwarded to the corresponding D-Wave NetworkX draw
function.

## Citation

For work that depends directly on this D-Wave integration, cite `DWave.jl`
using [`CITATION.cff`](CITATION.cff) or its
[Zenodo concept DOI](https://doi.org/10.5281/zenodo.7812066). The concept DOI is
the evergreen software identifier and resolves to the latest archived release.

For general discussion of the JuliaQUBO ecosystem, cite the
[QUBO.jl ecosystem article](https://doi.org/10.1080/10556788.2026.2702926).
When exact-release reproducibility matters, cite the corresponding Zenodo
version DOI instead; the archive for `v0.8.1` is
[10.5281/zenodo.23603558](https://doi.org/10.5281/zenodo.23603558).

## Development Checks
To verify that DWave can share a CondaPkg environment with the other
Python-backed JuliaQUBO benchmark drivers, run:

```sh
julia --startup-file=no test/shared_condapkg_resolve.jl
```

**Disclaimer:** _The D-Wave wrapper for Julia is not officially supported by D-Wave Systems. If you are a commercial customer interested in official support for Julia from D-Wave, let them know!_

**Note**: _If you are using [DWave.jl](https://github.com/JuliaQUBO/DWave.jl) in your project, we recommend you to include the `.CondaPkg` entry in your `.gitignore` file. The PythonCall module will place a lot of files in this folder when building its Python environment._
