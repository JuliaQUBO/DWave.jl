module TopologyWrapperTests

import DWave
import QUBOTools
import Test

const Graphs = DWave.Graphs

function _node_edge_pairs(arch)
    graph = QUBOTools.topology(QUBOTools.layout(arch))

    return Set(
        begin
            src = arch.nodes[Graphs.src(edge)]
            dst = arch.nodes[Graphs.dst(edge)]

            src <= dst ? (src, dst) : (dst, src)
        end for edge in Graphs.edges(graph)
    )
end

function _point_count(points)
    return length(unique(points))
end

Test.@testset "Pegasus layout uses Ocean topology data" begin
    arch = DWave.Pegasus(2)
    arch_layout = QUBOTools.layout(arch)
    graph = QUBOTools.topology(arch_layout)
    points = QUBOTools.geometry(arch_layout)

    Test.@test isdefined(DWave, :Pegasus)
    Test.@test DWave.layout(arch) isa QUBOTools.Layout
    Test.@test arch.size == 2
    Test.@test Graphs.nv(graph) == 40
    Test.@test Graphs.ne(graph) == 164
    Test.@test length(points) == 40
    Test.@test _point_count(points) > 4
    Test.@test _point_count(points[1:4]) == 4
    Test.@test arch.nodes[1] == 2
    Test.@test arch.node_indices[2] == 1
    Test.@test arch.coordinates[2] == (0, 0, 2, 0)
    Test.@test (2, 3) in _node_edge_pairs(arch)
end

Test.@testset "Zephyr layout uses Ocean topology data" begin
    arch = DWave.Zephyr(2)
    arch_layout = QUBOTools.layout(arch)
    graph = QUBOTools.topology(arch_layout)
    points = QUBOTools.geometry(arch_layout)

    Test.@test isdefined(DWave, :Zephyr)
    Test.@test DWave.layout(arch) isa QUBOTools.Layout
    Test.@test arch.size == 2
    Test.@test arch.shore_size == 4
    Test.@test Graphs.nv(graph) == 160
    Test.@test Graphs.ne(graph) == 1224
    Test.@test length(points) == 160
    Test.@test _point_count(points) > 4
    Test.@test _point_count(points[1:4]) == 4
    Test.@test arch.nodes[1] == 0
    Test.@test arch.node_indices[0] == 1
    Test.@test arch.coordinates[0] == (0, 0, 0, 0, 0)
    Test.@test (0, 1) in _node_edge_pairs(arch)
end

Test.@testset "WorkingGraph represents calibrated solver subsets" begin
    metadata = Dict{String,Any}(
        "dwave_info" => Dict{String,Any}(
            "chip_info" => Dict{String,Any}(
                "topology" => Dict{String,Any}("type" => "pegasus", "shape" => Any[2]),
                "qubits" => Any[2, 3, 28],
                "couplers" => Any[Any[2, 3], Any[2, 28]],
            ),
        ),
    )

    arch = DWave.WorkingGraph(metadata)
    arch_layout = QUBOTools.layout(arch)
    graph = QUBOTools.topology(arch_layout)
    points = QUBOTools.geometry(arch_layout)

    Test.@test isdefined(DWave, :WorkingGraph)
    Test.@test DWave.layout(arch) isa QUBOTools.Layout
    Test.@test arch.topology_type == "pegasus"
    Test.@test arch.topology_shape == [2]
    Test.@test arch.nodes == [2, 3, 28]
    Test.@test arch.coordinates[2] == (0, 0, 2, 0)
    Test.@test Graphs.nv(graph) == 3
    Test.@test Graphs.ne(graph) == 2
    Test.@test length(points) == 3
    Test.@test _point_count(points) == 3
    Test.@test (2, 3) in _node_edge_pairs(arch)
    Test.@test (2, 28) in _node_edge_pairs(arch)
    Test.@test !((3, 28) in _node_edge_pairs(arch))
end

if DWave.__auth__(; verbose = false)
    Test.@testset "WorkingGraph builds from live D-Wave sampler metadata" begin
        sampler = DWave.dwave_system.DWaveSampler(; token = ENV["DWAVE_API_TOKEN"])
        arch = DWave.WorkingGraph(sampler)
        arch_layout = QUBOTools.layout(arch)
        graph = QUBOTools.topology(arch_layout)
        points = QUBOTools.geometry(arch_layout)

        Test.@test arch.topology_type in ("pegasus", "zephyr")
        Test.@test !isempty(arch.topology_shape)
        Test.@test !isempty(arch.nodes)
        Test.@test !isempty(arch.edges)
        Test.@test Graphs.nv(graph) == length(arch.nodes)
        Test.@test Graphs.ne(graph) == length(arch.edges)
        Test.@test length(points) == length(arch.nodes)
        Test.@test length(arch.coordinates) == length(arch.nodes)
    end
end

Test.@testset "Topology wrappers validate inputs" begin
    Test.@test_throws ArgumentError DWave.Pegasus(0)
    Test.@test_throws ArgumentError DWave.Zephyr(0)
    Test.@test_throws ArgumentError DWave.Zephyr(2; shore_size = 0)
    Test.@test_throws ArgumentError DWave.WorkingGraph([1, 2], Any[Any[1, 3]])
    Test.@test_throws ArgumentError DWave.WorkingGraph(Dict{String,Any}("qubits" => Any[1, 2]))
end

end
