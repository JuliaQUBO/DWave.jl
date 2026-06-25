module ChimeraWrapperTests

import DWave
import QUBOTools
import Test

const Graphs = DWave.Graphs

function _edge_coordinate_pairs(arch::DWave.Chimera)
    coordinates = arch.coordinates

    return Set(
        begin
            src = coordinates[Graphs.src(edge)]
            dst = coordinates[Graphs.dst(edge)]

            src <= dst ? (src, dst) : (dst, src)
        end for edge in Graphs.edges(QUBOTools.topology(QUBOTools.layout(arch)))
    )
end

function _dnx_chimera_edge_coordinate_pairs(arch::DWave.Chimera)
    m, n = arch.grid_size
    shore_size = arch.cell_size ÷ 2
    graph = DWave.dwave_networkx.chimera_graph(m, n, shore_size; coordinates = true)
    edges = DWave.PythonCall.pyconvert(
        Vector{Tuple{NTuple{4,Int},NTuple{4,Int}}},
        DWave.PythonCall.pylist(graph.edges()),
    )

    return Set(src <= dst ? (src, dst) : (dst, src) for (src, dst) in edges)
end

Test.@testset "Chimera supports rectangular topology coordinates" begin
    arch = DWave.Chimera(2, 3)

    Test.@test isdefined(DWave, :Chimera)
    Test.@test arch.grid_size == (2, 3)
    Test.@test arch.cell_size == 8
    Test.@test length(arch.coordinates) == 2 * 3 * 8
    Test.@test arch.degree == 3
    Test.@test arch.effective_degree == 3
    Test.@test arch.coordinates[1] == (0, 0, 0, 0)
    Test.@test arch.coordinates[4] == (0, 0, 0, 3)
    Test.@test arch.coordinates[5] == (0, 0, 1, 0)
    Test.@test arch.coordinates[8] == (0, 0, 1, 3)
    Test.@test arch.coordinates[9] == (0, 1, 0, 0)
    Test.@test arch.coordinates[41] == (1, 2, 0, 0)
    Test.@test arch.coordinates[48] == (1, 2, 1, 3)
end

Test.@testset "Chimera layout returns topology graph and geometry" begin
    arch = DWave.Chimera(2, 3)
    arch_layout = QUBOTools.layout(arch)
    graph = QUBOTools.topology(arch_layout)
    points = QUBOTools.geometry(arch_layout)

    Test.@test QUBOTools.layout(arch) isa QUBOTools.Layout
    Test.@test Graphs.nv(graph) == 48
    Test.@test Graphs.ne(graph) == 124
    Test.@test length(points) == 48
    Test.@test points[1] == QUBOTools.Point{2,Float64}(0.0, -0.0625)
    Test.@test points[5] == QUBOTools.Point{2,Float64}(-0.0625, 0.0)

    edge_pairs = _edge_coordinate_pairs(arch)

    Test.@test ((0, 0, 0, 0), (0, 0, 1, 0)) in edge_pairs
    Test.@test ((0, 0, 0, 0), (0, 0, 1, 3)) in edge_pairs
    Test.@test ((0, 0, 0, 0), (1, 0, 0, 0)) in edge_pairs
    Test.@test ((0, 0, 1, 0), (0, 1, 1, 0)) in edge_pairs
    Test.@test !(((0, 0, 0, 0), (0, 1, 0, 0)) in edge_pairs)
    Test.@test !(((0, 0, 1, 0), (1, 0, 1, 0)) in edge_pairs)
end

Test.@testset "Chimera topology matches dwave-networkx coordinate graph" begin
    arch = DWave.Chimera(2, 3)

    Test.@test _edge_coordinate_pairs(arch) == _dnx_chimera_edge_coordinate_pairs(arch)
end

Test.@testset "Chimera validates dimensions" begin
    Test.@test_throws ArgumentError DWave.Chimera(0, 2)
    Test.@test_throws ArgumentError DWave.Chimera(2, 0)
    Test.@test_throws ArgumentError DWave.Chimera(2, 2; cell_size = 7)
    Test.@test_throws ArgumentError DWave.Chimera(2, 3; degree = 2)
end

Test.@testset "Chimera device scales coefficients into integer backend" begin
    model = QUBOTools.Model{Symbol,Float64,Int}(
        Dict(:a => 0.5),
        Dict((:a, :b) => -1.0);
        scale = 2.0,
        offset = 0.25,
        domain = :spin,
        start = Dict(:a => 1, :b => -1),
    )
    dev = DWave.DWaveDevice(DWave.Chimera(1, 1), model)
    backend = QUBOTools.backend(dev)

    Test.@test isdefined(DWave, :DWaveDevice)
    Test.@test dev.factor == 200_000.0
    Test.@test Dict(QUBOTools.linear_terms(backend)) == Dict(1 => 50_000)
    Test.@test Dict(QUBOTools.quadratic_terms(backend)) == Dict((1, 2) => -100_000)
    Test.@test QUBOTools.scale(backend) == 1
    Test.@test QUBOTools.offset(backend) == 25_000
    Test.@test QUBOTools.start(backend) == Dict(1 => 1, 2 => -1)
end

end
