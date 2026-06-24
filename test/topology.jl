module TopologyWrapperTests

import DWave
import QUBOTools
import Test

const Graphs = DWave.Graphs

abstract type DWaveArchitecture <: QUBOTools.AbstractArchitecture end

include(joinpath(@__DIR__, "..", "src", "wrapper", "topology.jl"))

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

Test.@testset "Pegasus layout uses Ocean topology data" begin
    arch = Pegasus(2)
    arch_layout = QUBOTools.layout(arch)
    graph = QUBOTools.topology(arch_layout)
    points = QUBOTools.geometry(arch_layout)

    Test.@test layout(arch) isa QUBOTools.Layout
    Test.@test arch.size == 2
    Test.@test Graphs.nv(graph) == 40
    Test.@test Graphs.ne(graph) == 164
    Test.@test length(points) == 40
    Test.@test arch.nodes[1] == 2
    Test.@test arch.node_indices[2] == 1
    Test.@test arch.coordinates[2] == (0, 0, 2, 0)
    Test.@test (2, 3) in _node_edge_pairs(arch)
end

Test.@testset "Zephyr layout uses Ocean topology data" begin
    arch = Zephyr(2)
    arch_layout = QUBOTools.layout(arch)
    graph = QUBOTools.topology(arch_layout)
    points = QUBOTools.geometry(arch_layout)

    Test.@test layout(arch) isa QUBOTools.Layout
    Test.@test arch.size == 2
    Test.@test arch.shore_size == 4
    Test.@test Graphs.nv(graph) == 160
    Test.@test Graphs.ne(graph) == 1224
    Test.@test length(points) == 160
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

    arch = WorkingGraph(metadata)
    arch_layout = QUBOTools.layout(arch)
    graph = QUBOTools.topology(arch_layout)
    points = QUBOTools.geometry(arch_layout)

    Test.@test layout(arch) isa QUBOTools.Layout
    Test.@test arch.topology_type == "pegasus"
    Test.@test arch.topology_shape == [2]
    Test.@test arch.nodes == [2, 3, 28]
    Test.@test arch.coordinates[2] == (0, 0, 2, 0)
    Test.@test Graphs.nv(graph) == 3
    Test.@test Graphs.ne(graph) == 2
    Test.@test length(points) == 3
    Test.@test (2, 3) in _node_edge_pairs(arch)
    Test.@test (2, 28) in _node_edge_pairs(arch)
    Test.@test !((3, 28) in _node_edge_pairs(arch))
end

Test.@testset "Topology wrappers validate inputs" begin
    Test.@test_throws ArgumentError Pegasus(0)
    Test.@test_throws ArgumentError Zephyr(0)
    Test.@test_throws ArgumentError Zephyr(2; shore_size = 0)
    Test.@test_throws ArgumentError WorkingGraph([1, 2], Any[Any[1, 3]])
    Test.@test_throws ArgumentError WorkingGraph(Dict{String,Any}("qubits" => Any[1, 2]))
end

end
