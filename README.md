# DWave.jl
[![QUBODRIVERS](https://img.shields.io/badge/Powered%20by-QUBODrivers.jl-%20%234063d8)](https://github.com/psrenergy/QUBODrivers.jl)

D-Wave Quantum Annealing Interface for JuMP

## Installation
```julia
julia> import Pkg

julia> Pkg.add("DWave")
```

## Migration from DWaveNeal.jl
`DWaveNeal.jl` is deprecated and kept only as a compatibility shim. New code
should use `DWave.jl` directly:

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

## API Token
To use D-Wave's QPU it is necessary to obtain an API Token from [Leap](https://cloud.dwavesys.com/leap/).

**Disclaimer:** _The D-Wave wrapper for Julia is not officially supported by D-Wave Systems. If you are a commercial customer interested in official support for Julia from D-Wave, let them know!_

**Note**: _If you are using [DWave.jl](https://github.com/psrenergy/DWave.jl) in your project, we recommend you to include the `.CondaPkg` entry in your `.gitignore` file. The PythonCall module will place a lot of files in this folder when building its Python environment._
