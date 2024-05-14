# DWave.jl
[![QUBODRIVERS](https://img.shields.io/badge/Powered%20by-QUBODrivers.jl-%20%234063d8)](https://github.com/psrenergy/QUBODrivers.jl)

D-Wave Quantum Annealing Interface for JuMP

## Installation
```julia
julia> import Pkg

julia> Pkg.add("DWave.jl")
```

## Basic Usage
```julia
using JuMP
using QUBODrivers
using DWave

model = Model(DWave.Optimizer)

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

## API Token
To use D-Wave's QPU it is necessary to obtain an API Token from [Leap](https://cloud.dwavesys.com/leap/).

**Disclaimer:** _The D-Wave wrapper for Julia is not officially supported by D-Wave Systems. If you are a commercial customer interested in official support for Julia from D-Wave, let them know!_

**Note**: _If you are using [DWave.jl](https://github.com/psrenergy/DWave.jl) in your project, we recommend you to include the `.CondaPkg` entry in your `.gitignore` file. The PythonCall module will place a lot of files in this folder when building its Python environment._
