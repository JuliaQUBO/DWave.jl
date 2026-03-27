import DWave
import QUBODrivers
import Random

if DWave.__auth__(; verbose = false)
    QUBODrivers.test(DWave.Optimizer; examples = true)
else
    @warn "DWave.Optimizer tests skipped since API Token is missing."
end

QUBODrivers.test(DWave.Neal.Optimizer; examples = true)

include("neal_parity.jl")
