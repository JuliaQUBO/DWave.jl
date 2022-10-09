# DWave.jl
Wrapper for the D-Wave Quantum API (ft. [Anneal.jl](https://github.com/psrenergy/Anneal.jl))

## Installation
```julia
julia> import Pkg

julia> Pkg.add(url="https://github.com/psrenergy/DWave.jl#main")
```

**Disclaimer:** _The D-Wave wrapper for Julia is not officially supported by D-Wave Systems. If you are a commercial customer interested in official support for Julia from D-Wave, let them know!_

**Note**: _If you are using [DWave.jl](https://github.com/psrenergy/DWave.jl) in your project, we recommend you to include the `.CondaPkg` entry in your `.gitignore` file. The PythonCall module will place a lot of files in this folder when building its Python environment._
