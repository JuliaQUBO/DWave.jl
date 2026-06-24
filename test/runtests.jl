import DWave
import QUBODrivers

include("compat_metadata.jl")
include("auth.jl")

if DWave.__auth__(; verbose = false)
    QUBODrivers.test(DWave.Optimizer; examples = true, benchmark_conformance = true)
else
    @warn "DWave.Optimizer tests skipped since API Token is missing."
end

QUBODrivers.test(DWave.Neal.Optimizer; examples = true, benchmark_conformance = true)
QUBODrivers.test(DWave.Greedy.Optimizer; examples = true, benchmark_conformance = true)
QUBODrivers.test(DWave.Random.Optimizer; examples = true, benchmark_conformance = true)
QUBODrivers.test(DWave.Tabu.Optimizer; examples = true, benchmark_conformance = true)

include("neal_parity.jl")
include("classical_parity.jl")
include("dwave_metadata.jl")
include("chimera.jl")
