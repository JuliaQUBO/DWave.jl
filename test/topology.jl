module TopologyWrapperTests

import DWave
import QUBOTools
import Test

const Graphs = DWave.Graphs
const PythonCall = DWave.PythonCall

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

function _py_len(value)
    return PythonCall.pyconvert(Int, PythonCall.pybuiltins.len(value))
end

function _py_class_name(value)
    return PythonCall.pyconvert(String, value.__class__.__name__)
end

function _close_figure(figure)
    PythonCall.pyimport("matplotlib.pyplot").close(figure)

    return nothing
end

function _use_matplotlib_test_backend()
    config_dir = get!(ENV, "MPLCONFIGDIR") do
        mktempdir()
    end
    PythonCall.pyimport("os").environ["MPLCONFIGDIR"] = config_dir
    PythonCall.pyimport("matplotlib").use("Agg"; force = true)

    return nothing
end

Test.@testset "Pegasus layout uses Ocean topology data" begin
    arch = DWave.Pegasus(2)
    arch_layout = QUBOTools.layout(arch)
    graph = QUBOTools.topology(arch_layout)
    points = QUBOTools.geometry(arch_layout)

    Test.@test isdefined(DWave, :Pegasus)
    Test.@test QUBOTools.layout(arch) isa QUBOTools.Layout
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
    Test.@test QUBOTools.layout(arch) isa QUBOTools.Layout
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
    Test.@test QUBOTools.layout(arch) isa QUBOTools.Layout
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

Test.@testset "WorkingGraph unknown topology uses graph geometry fallback" begin
    arch = DWave.WorkingGraph(
        [1, 2, 3],
        Any[Any[1, 2], Any[2, 3]];
        topology = Dict{String,Any}("type" => "custom", "shape" => Any[1]),
        coordinates = Dict{Int,Any}(
            1 => (0, 0, 0),
            2 => (0, 0, 1),
            3 => (0, 1, 0),
        ),
    )
    arch_layout = QUBOTools.layout(arch)
    points = QUBOTools.geometry(arch_layout)

    Test.@test length(points) == 3
    Test.@test _point_count(points) == 3
end

Test.@testset "D-Wave NetworkX drawing helpers use full working graphs" begin
    _use_matplotlib_test_backend()

    pegasus_metadata = Dict{String,Any}(
        "dwave_info" => Dict{String,Any}(
            "chip_info" => Dict{String,Any}(
                "topology" => Dict{String,Any}("type" => "pegasus", "shape" => Any[2]),
                "qubits" => Any[2, 3, 28],
                "couplers" => Any[Any[2, 3], Any[2, 28]],
            ),
            "embedding_context" => Dict{String,Any}(
                "embedding" => Dict{Any,Any}(1 => Any[2], 2 => Any[28]),
            ),
        ),
    )
    pegasus = DWave.WorkingGraph(pegasus_metadata)
    pegasus_graph = DWave._dnx_hardware_graph(pegasus)

    Test.@test isdefined(DWave, :draw_topology)
    Test.@test isdefined(DWave, :draw_embedding)
    Test.@test _py_len(pegasus_graph.nodes()) == length(pegasus.nodes)
    Test.@test _py_len(pegasus_graph.edges()) == length(pegasus.edges)

    figure = DWave.draw_topology(pegasus; node_size = 8, with_labels = false)

    try
        Test.@test _py_class_name(figure) == "Figure"
    finally
        _close_figure(figure)
    end

    figure = DWave.draw_embedding(pegasus_metadata; node_size = 8, with_labels = false)

    try
        Test.@test _py_class_name(figure) == "Figure"
    finally
        _close_figure(figure)
    end

    zephyr_metadata = Dict{String,Any}(
        "dwave_info" => Dict{String,Any}(
            "chip_info" => Dict{String,Any}(
                "topology" => Dict{String,Any}("type" => "zephyr", "shape" => Any[2, 4]),
                "qubits" => Any[0, 1, 4],
                "couplers" => Any[Any[0, 1], Any[1, 4]],
            ),
            "embedding_context" => Dict{String,Any}(
                "embedding" => Dict{Any,Any}(1 => Any[0], 2 => Any[4]),
            ),
        ),
    )
    zephyr = DWave.WorkingGraph(zephyr_metadata)
    zephyr_graph = DWave._dnx_hardware_graph(zephyr)

    Test.@test _py_len(zephyr_graph.nodes()) == length(zephyr.nodes)
    Test.@test _py_len(zephyr_graph.edges()) == length(zephyr.edges)

    figure = DWave.draw_topology(zephyr; node_size = 8, with_labels = false)

    try
        Test.@test _py_class_name(figure) == "Figure"
    finally
        _close_figure(figure)
    end

    figure = DWave.draw_embedding(zephyr, Dict(1 => [0], 2 => [4]); node_size = 8, with_labels = false)

    try
        Test.@test _py_class_name(figure) == "Figure"
    finally
        _close_figure(figure)
    end

    Test.@test_throws ArgumentError DWave.draw_embedding(Dict{String,Any}(
        "dwave_info" => Dict{String,Any}(
            "chip_info" => pegasus_metadata["dwave_info"]["chip_info"],
        ),
    ))
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
