# D-Wave

This folder contains interface definitions related to D-Wave hardware.

`Chimera`, `Pegasus`, and `Zephyr` describe ideal hardware families for offline
topology work. `WorkingGraph` describes the calibrated node and coupler subset
reported by a live solver; use it when solver metadata includes unavailable
qubits or couplers removed by calibration.
