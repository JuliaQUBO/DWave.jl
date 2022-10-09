using Test
using DWave
using Anneal

function main()
    Anneal.test(DWave.Optimizer)
end

main() # Here we go!