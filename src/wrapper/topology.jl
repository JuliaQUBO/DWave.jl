# NOTE: These wrappers describe D-Wave hardware graphs as QUBOTools layouts.
# Use Pegasus and Zephyr for ideal offline topology work; use WorkingGraph when
# solver metadata provides the calibrated node and coupler subset for a live QPU.

abstract type DWaveHardwareTopology <: DWaveArchitecture end

struct Pegasus <: DWaveHardwareTopology
    size::Int
    nodes::Vector{Int}
    edges::Vector{Tuple{Int,Int}}
    coordinates::Dict{Int,NTuple{4,Int}}
    node_indices::Dict{Int,Int}
end

struct Zephyr <: DWaveHardwareTopology
    size::Int
    shore_size::Int
    nodes::Vector{Int}
    edges::Vector{Tuple{Int,Int}}
    coordinates::Dict{Int,NTuple{5,Int}}
    node_indices::Dict{Int,Int}
end

struct WorkingGraph <: DWaveHardwareTopology
    topology_type::Union{String,Nothing}
    topology_shape::Vector{Int}
    nodes::Vector{Int}
    edges::Vector{Tuple{Int,Int}}
    coordinates::Dict{Int,Tuple}
    node_indices::Dict{Int,Int}
    metadata::Dict{String,Any}
end

function Pegasus(m::Integer = 16)
    m > 0 || throw(ArgumentError("Pegasus size must be positive, got $m"))

    graph = _dnx().pegasus_graph(Int(m))
    nodes, edges = _dnx_nodes_edges(graph)
    coordinates = _pegasus_coordinates(Int(m), nodes)

    return Pegasus(Int(m), nodes, edges, coordinates, _node_indices(nodes))
end

function Zephyr(m::Integer = 4; shore_size::Integer = 4)
    m > 0 || throw(ArgumentError("Zephyr size must be positive, got $m"))
    shore_size > 0 || throw(ArgumentError("Zephyr shore size must be positive, got $shore_size"))

    graph = _dnx().zephyr_graph(Int(m), Int(shore_size))
    nodes, edges = _dnx_nodes_edges(graph)
    coordinates = _zephyr_coordinates(Int(m), Int(shore_size), nodes)

    return Zephyr(Int(m), Int(shore_size), nodes, edges, coordinates, _node_indices(nodes))
end

function WorkingGraph(
    nodes::AbstractVector,
    edges::AbstractVector;
    topology = nothing,
    coordinates::Union{AbstractDict,Nothing} = nothing,
    metadata::AbstractDict = Dict{String,Any}(),
)
    normal_nodes = _normalise_nodes(nodes)
    normal_edges = _normalise_edges(edges)
    node_indices = _node_indices(normal_nodes)

    for (u, v) in normal_edges
        if !haskey(node_indices, u) || !haskey(node_indices, v)
            throw(ArgumentError("working graph edge ($u, $v) references a node outside the node list"))
        end
    end

    topology_type, topology_shape = _normalise_topology_metadata(topology)
    normal_coordinates = if coordinates === nothing
        _metadata_topology_coordinates(topology_type, topology_shape, normal_nodes)
    else
        _normalise_coordinates(coordinates)
    end

    return WorkingGraph(
        topology_type,
        topology_shape,
        normal_nodes,
        normal_edges,
        normal_coordinates,
        node_indices,
        Dict{String,Any}(string(k) => v for (k, v) in pairs(metadata)),
    )
end

function WorkingGraph(metadata::AbstractDict)
    chip_info = _metadata_chip_info(metadata)
    nodes = _metadata_first(chip_info, ("qubits", "nodes", "node_list", "nodelist"))
    edges = _metadata_first(chip_info, ("couplers", "edges", "edge_list", "edgelist"))

    nodes === nothing && throw(ArgumentError("working graph metadata is missing a qubits/node list"))
    edges === nothing && throw(ArgumentError("working graph metadata is missing a couplers/edge list"))

    return WorkingGraph(
        nodes,
        edges;
        topology = get(chip_info, "topology", nothing),
        metadata = chip_info,
    )
end

function WorkingGraph(dwave_sampler::PythonCall.Py)
    chip_info = _dwave_chip_info(dwave_sampler)
    isempty(chip_info) && throw(ArgumentError("D-Wave sampler does not expose working graph metadata"))

    return WorkingGraph(chip_info)
end

function _dnx()
    return dwave_networkx
end

function _pylist(value)
    return PythonCall.pylist(value)
end

function _pyconvert(::Type{T}, value) where {T}
    return PythonCall.pyconvert(T, value)
end

function _py(value)
    return PythonCall.Py(value)
end

function _dnx_nodes_edges(graph)
    nodes = sort(_pyconvert(Vector{Int}, _pylist(graph.nodes())))
    edges = _normalise_edges(_pyconvert(Vector{Tuple{Int,Int}}, _pylist(graph.edges())))

    return nodes, edges
end

function _node_indices(nodes::Vector{Int})
    return Dict(node => i for (i, node) in pairs(nodes))
end

function _normalise_nodes(nodes::AbstractVector)
    normal_nodes = Int[]
    sizehint!(normal_nodes, length(nodes))

    for node in nodes
        node isa Integer || throw(ArgumentError("working graph node '$node' is not an integer"))

        push!(normal_nodes, Int(node))
    end

    return sort!(unique(normal_nodes))
end

function _normalise_edge(edge)
    if edge isa Tuple || edge isa AbstractVector
        length(edge) == 2 || throw(ArgumentError("working graph edge '$edge' must have two endpoints"))

        u, v = edge
        u isa Integer || throw(ArgumentError("working graph edge endpoint '$u' is not an integer"))
        v isa Integer || throw(ArgumentError("working graph edge endpoint '$v' is not an integer"))
        u = Int(u)
        v = Int(v)
        u == v && throw(ArgumentError("working graph self edge ($u, $v) is not supported"))

        return u <= v ? (u, v) : (v, u)
    end

    throw(ArgumentError("working graph edge '$edge' must be a tuple or vector"))
end

function _normalise_edges(edges::AbstractVector)
    normal_edges = Set{Tuple{Int,Int}}()

    for edge in edges
        push!(normal_edges, _normalise_edge(edge))
    end

    return sort!(collect(normal_edges))
end

function _normalise_coordinates(coordinates::AbstractDict)
    normal_coordinates = Dict{Int,Tuple}()

    for (node, coordinate) in pairs(coordinates)
        node isa Integer || throw(ArgumentError("coordinate node '$node' is not an integer"))
        coordinate isa Tuple || coordinate isa AbstractVector ||
            throw(ArgumentError("coordinate '$coordinate' must be a tuple or vector"))

        normal_coordinates[Int(node)] = tuple((Int(x) for x in coordinate)...)
    end

    return normal_coordinates
end

function _pegasus_coordinates(m::Int, nodes::AbstractVector{Int})
    converter = _dnx().pegasus_coordinates(m)

    return Dict{Int,NTuple{4,Int}}(
        node => _pyconvert(NTuple{4,Int}, converter.linear_to_pegasus(node))
        for node in nodes
    )
end

function _zephyr_coordinates(m::Int, shore_size::Int, nodes::AbstractVector{Int})
    converter = _dnx().zephyr_coordinates(m, shore_size)

    return Dict{Int,NTuple{5,Int}}(
        node => _pyconvert(NTuple{5,Int}, converter.linear_to_zephyr(node))
        for node in nodes
    )
end

function _metadata_topology_coordinates(
    topology_type::Union{String,Nothing},
    topology_shape::Vector{Int},
    nodes::AbstractVector{Int},
)
    topology_type === nothing && return Dict{Int,Tuple}()
    isempty(topology_shape) && return Dict{Int,Tuple}()

    coordinates = Dict{Int,Tuple}()

    if topology_type == "pegasus"
        converter = _dnx().pegasus_coordinates(topology_shape[1])

        for node in nodes
            try
                coordinates[node] = _pyconvert(Tuple, converter.linear_to_pegasus(node))
            catch
            end
        end
    elseif topology_type == "zephyr"
        shore_size = length(topology_shape) >= 2 ? topology_shape[2] : 4
        converter = _dnx().zephyr_coordinates(topology_shape[1], shore_size)

        for node in nodes
            try
                coordinates[node] = _pyconvert(Tuple, converter.linear_to_zephyr(node))
            catch
            end
        end
    end

    return coordinates
end

function _normalise_topology_metadata(topology)
    topology === nothing && return nothing, Int[]

    if topology isa AbstractString
        return lowercase(String(topology)), Int[]
    end

    if topology isa AbstractDict
        topology_type = get(topology, "type", get(topology, :type, nothing))
        shape = get(topology, "shape", get(topology, :shape, Int[]))

        normal_type = topology_type === nothing ? nothing : lowercase(String(topology_type))
        normal_shape = _normalise_topology_shape(shape)

        return normal_type, normal_shape
    end

    throw(ArgumentError("topology metadata must be a string, dictionary, or nothing"))
end

function _normalise_topology_shape(shape)
    shape === nothing && return Int[]
    shape isa Integer && return [Int(shape)]
    shape isa AbstractVector || return Int[]

    return Int[Int(x) for x in shape if x isa Integer]
end

function _metadata_chip_info(metadata::AbstractDict)
    if haskey(metadata, "dwave_info")
        dwave_info = metadata["dwave_info"]

        if dwave_info isa AbstractDict && haskey(dwave_info, "chip_info")
            chip_info = dwave_info["chip_info"]
            chip_info isa AbstractDict && return chip_info
        end
    end

    if haskey(metadata, "chip_info")
        chip_info = metadata["chip_info"]
        chip_info isa AbstractDict && return chip_info
    end

    return metadata
end

function _metadata_first(metadata::AbstractDict, keys)
    for key in keys
        haskey(metadata, key) && return metadata[key]
    end

    return nothing
end

function _topology_graph(nodes::Vector{Int}, edges::Vector{Tuple{Int,Int}}, node_indices::Dict{Int,Int})
    graph = Graphs.SimpleGraph(length(nodes))

    for (u, v) in edges
        Graphs.add_edge!(graph, node_indices[u], node_indices[v])
    end

    return graph
end

function _dnx_working_graph(arch::WorkingGraph)
    isempty(arch.topology_shape) && return nothing

    node_list = _py(arch.nodes)
    edge_list = _py(arch.edges)

    if arch.topology_type == "pegasus"
        return _dnx().pegasus_graph(
            arch.topology_shape[1];
            node_list = node_list,
            edge_list = edge_list,
        )
    elseif arch.topology_type == "zephyr"
        shore_size = length(arch.topology_shape) >= 2 ? arch.topology_shape[2] : 4

        return _dnx().zephyr_graph(
            arch.topology_shape[1],
            shore_size;
            node_list = node_list,
            edge_list = edge_list,
        )
    end

    return nothing
end

function _dnx_layout_points(layout, nodes::Vector{Int})
    points = Vector{QUBOTools.Point{2,Float64}}(undef, length(nodes))

    for (index, node) in pairs(nodes)
        position = _pyconvert(Tuple, layout[node])

        points[index] = QUBOTools.Point{2,Float64}(
            Float64(position[1]),
            Float64(position[2]),
        )
    end

    return points
end

function _pegasus_points(arch::Pegasus)
    graph = _dnx().pegasus_graph(arch.size)

    return _dnx_layout_points(_dnx().pegasus_layout(graph), arch.nodes)
end

function _zephyr_points(arch::Zephyr)
    graph = _dnx().zephyr_graph(arch.size, arch.shore_size)

    return _dnx_layout_points(_dnx().zephyr_layout(graph), arch.nodes)
end

function _working_graph_layout_points(arch::WorkingGraph)
    graph = _dnx_working_graph(arch)
    graph === nothing && return nothing

    if arch.topology_type == "pegasus"
        return _dnx_layout_points(_dnx().pegasus_layout(graph), arch.nodes)
    elseif arch.topology_type == "zephyr"
        return _dnx_layout_points(_dnx().zephyr_layout(graph), arch.nodes)
    end

    return nothing
end

function _coordinate_point(coordinate::Tuple)
    if length(coordinate) >= 2
        return QUBOTools.Point{2,Float64}(Float64(coordinate[2]), Float64(coordinate[1]))
    end

    return QUBOTools.Point{2,Float64}(0.0, 0.0)
end

function _coordinate_points(nodes::Vector{Int}, coordinates::AbstractDict)
    points = Vector{QUBOTools.Point{2,Float64}}(undef, length(nodes))

    for (index, node) in pairs(nodes)
        points[index] = _coordinate_point(coordinates[node])
    end

    return points
end

function _working_graph_points(arch::WorkingGraph, graph)
    points = _working_graph_layout_points(arch)
    points === nothing || return points

    if all(node -> haskey(arch.coordinates, node), arch.nodes)
        return _coordinate_points(arch.nodes, arch.coordinates)
    end

    return QUBOTools.geometry(graph)
end

function QUBOTools.topology(arch::Union{Pegasus,Zephyr,WorkingGraph})
    return _topology_graph(arch.nodes, arch.edges, arch.node_indices)
end

function QUBOTools.geometry(arch::Pegasus)
    return _pegasus_points(arch)
end

function QUBOTools.geometry(arch::Zephyr)
    return _zephyr_points(arch)
end

function QUBOTools.layout(arch::Union{Pegasus,Zephyr})
    return QUBOTools.Layout(QUBOTools.topology(arch), QUBOTools.geometry(arch))
end

function QUBOTools.geometry(arch::WorkingGraph)
    return _working_graph_points(arch, QUBOTools.topology(arch))
end

function QUBOTools.layout(arch::WorkingGraph)
    graph = QUBOTools.topology(arch)

    return QUBOTools.Layout(graph, _working_graph_points(arch, graph))
end

function layout(arch::DWaveHardwareTopology)
    return QUBOTools.layout(arch)
end
