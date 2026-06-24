# NOTE: This file is temporary. In the future, it should be moved to the
# DWave.jl package, so that this functionality is activated only given a
# specific context of use.

@doc raw"""
    Chimera
    
The format of the instance-description file starts with a line giving the size of the Chimera graph[^alex1770].
Two numbers are given to specify an ``m \times n`` rectangle.

The subsequent lines are of the form
```
    <Chimera vertex> <Chimera vertex> <weight>
```
where `<Chimera vertex>` is specified by four numbers using the format, Chimera graph, ``C_N``:
    
Vertices are ``v = (x, y, o, i)`` where
- ``x, y \in [0, N - 1]`` are the horizontal, vertical coordinates of the ``K_{4, 4}``
- ``o \in [0, 1]`` is the **orientation**: (``0 = \text{horizontally connected}``, ``1 = \text{vertically connected}``)
- ``i \in [0, 3]`` is the index within the "semi-``K_{4,4}``" or "bigvertex"
- There is an involution given by ``x \iff y``, ``o \iff 1 - o``

There is an edge from ``v_p`` to ``v_q`` if at least one of the following holds:
- ``(x_p, y_p) = (x_q, y_q) \wedge o_p \neq o_q``
- ``|x_p - x_q| = 1 \wedge y_p = y_q \wedge o_p = o_q = 0 \wedge i_p = i_q``
- ``x_p = x_q \wedge |y_p-y_q| = 1 \wedge o_p = o_q = 1 \wedge i_p = i_q``

[^alex1770]:
    `alex1770`'s QUBO-Chimera Git Repository [{git}](https://github.com/alex1770/QUBO-Chimera)

[^dwave]:
    **D-Wave QPU Architecture: Topologies** [{docs}](https://docs.dwavesys.com/docs/latest/c_gs_4.html)
"""
struct Chimera <: DWaveArchitecture
    grid_size::NTuple{2,Int}
    cell_size::Int
    precision::Int
    coordinates::Dict{Int,NTuple{4,Int}}
    degree::Int
    effective_degree::Int
end

function Chimera(
    m::Integer = 16, # 2000Q
    n::Integer = m;
    cell_size::Integer = 8,
    precision::Integer = 5,
    degree::Union{Integer,Nothing} = nothing, 
)
    m > 0 || throw(ArgumentError("Chimera row count must be positive, got $m"))
    n > 0 || throw(ArgumentError("Chimera column count must be positive, got $n"))
    cell_size > 0 || throw(ArgumentError("Chimera cell size must be positive, got $cell_size"))
    iseven(cell_size) || throw(ArgumentError("Chimera cell size must be even, got $cell_size"))

    grid_size = (m, n)
    dimension = m * n * cell_size

    min_degree = max(m, n)

    if isnothing(degree)
        degree = min_degree
    end

    if degree < min_degree
        throw(ArgumentError(
            """
            degree of '$(degree)' was specified.
            However, the minimum degree required for a system with '$(dimension)' sites is '$(min_degree)'.
            """,
        ))
    end

    shore_size = cell_size ÷ 2
    coordinates = Dict{Int,NTuple{4,Int}}()
    sizehint!(coordinates, dimension)

    for x in 0:(m - 1), y in 0:(n - 1), o in 0:1, i in 0:(shore_size - 1)
        coordinates[_chimera_index(x, y, o, i, n, shore_size)] = (x, y, o, i)
    end

    effective_degree = 1 + maximum(c -> max(c[1], c[2]), values(coordinates))

    if effective_degree > degree
        throw(ArgumentError(
            """
            The effective degree '$effective_degree' value is greater than the chimera degree '$(degree)', which is infeasible.
            """,
        ))
    end

    return Chimera(
        grid_size,
        cell_size,
        precision,
        coordinates,
        degree,
        effective_degree,
    )
end

function _chimera_index(
    x::Integer,
    y::Integer,
    o::Integer,
    i::Integer,
    n::Integer,
    shore_size::Integer,
)
    return (((x * n + y) * 2 + o) * shore_size + i) + 1
end

function _variable_starts(model::QUBOTools.AbstractModel{V}) where {V}
    return Dict{V,Int}(
        QUBOTools.variable(model, i) => Int(v)
        for (i, v) in QUBOTools.start(model)
    )
end

function DWaveDevice(arch::Chimera, model::QUBOTools.AbstractModel{V}) where {V}
    L = collect(QUBOTools.linear_terms(model))
    Q = collect(QUBOTools.quadratic_terms(model))
    α = QUBOTools.scale(model)
    β = QUBOTools.offset(model)

    γ = max(
        isempty(L) ? 0 : maximum(abs(v) for (_, v) in L),
        isempty(Q) ? 0 : maximum(abs(v) for (_, v) in Q),
    )

    γχ = iszero(γ) ? 1.0 : 10.0 ^ arch.precision / γ

    # Chimera Coefficients
    Lχ = sizehint!(Dict{V,Int}(), length(L))
    Qχ = sizehint!(Dict{Tuple{V,V},Int}(), length(Q))

    for (i, v) in L
        xi = QUBOTools.variable(model, i) 

        Lχ[xi] = round(Int, v * γχ)
    end

    for ((i, j), v) in Q
        xi = QUBOTools.variable(model, i)
        xj = QUBOTools.variable(model, j)

        Qχ[(xi, xj)] = round(Int, v * γχ)
    end

    αχ = 1
    βχ = round(Int, β * γχ)

    factor = α * γχ

    return DWaveDevice(
        arch,
        QUBOTools.Model{V,Int,Int}(
            Set{V}(QUBOTools.variables(model)),
            Lχ,
            Qχ;
            scale    = αχ,
            offset   = βχ,
            sense    = QUBOTools.sense(model),
            domain   = QUBOTools.domain(model),
            metadata = copy(QUBOTools.metadata(model)),
            start    = _variable_starts(model),
        ),
        factor,
    )
end

function _chimera_graph(arch::Chimera)
    m, n = arch.grid_size
    shore_size = arch.cell_size ÷ 2
    graph = Graphs.SimpleGraph(length(arch.coordinates))

    for x in 0:(m - 1), y in 0:(n - 1)
        for i in 0:(shore_size - 1), j in 0:(shore_size - 1)
            Graphs.add_edge!(
                graph,
                _chimera_index(x, y, 0, i, n, shore_size),
                _chimera_index(x, y, 1, j, n, shore_size),
            )
        end

        if x < m - 1
            for i in 0:(shore_size - 1)
                Graphs.add_edge!(
                    graph,
                    _chimera_index(x, y, 0, i, n, shore_size),
                    _chimera_index(x + 1, y, 0, i, n, shore_size),
                )
            end
        end

        if y < n - 1
            for i in 0:(shore_size - 1)
                Graphs.add_edge!(
                    graph,
                    _chimera_index(x, y, 1, i, n, shore_size),
                    _chimera_index(x, y + 1, 1, i, n, shore_size),
                )
            end
        end
    end

    return graph
end

function _chimera_points(arch::Chimera)
    shore_size = arch.cell_size ÷ 2
    cell_spacing = 1.0 / arch.effective_degree
    shore_spacing = cell_spacing / (2 * shore_size)
    points = Vector{QUBOTools.Point{2,Float64}}(undef, length(arch.coordinates))

    for (index, (x, y, o, i)) in arch.coordinates
        offset = (i - (shore_size - 1) / 2) * shore_spacing

        points[index] = if o == 0
            QUBOTools.Point{2,Float64}(y, x + offset)
        else
            QUBOTools.Point{2,Float64}(y + offset, x)
        end
    end

    return points
end

function QUBOTools.topology(arch::Chimera)
    return _chimera_graph(arch)
end

function QUBOTools.geometry(arch::Chimera)
    return _chimera_points(arch)
end

function QUBOTools.layout(arch::Chimera)
    return QUBOTools.Layout(QUBOTools.topology(arch), QUBOTools.geometry(arch))
end

function layout(arch::Chimera)
    return QUBOTools.layout(arch)
end
