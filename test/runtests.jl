using DWave: DWave, QUBO.QUBODrivers

# QUBODrivers.test(DWave.Optimizer; examples = true)
QUBODrivers.test(DWave.Neal.Optimizer; examples = true)
