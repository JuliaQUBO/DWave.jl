import Pkg

const PACKAGE_ROOT = normpath(joinpath(@__DIR__, ".."))

function _package_version(name::String)
    versions = [
        dep.version for dep in values(Pkg.dependencies()) if dep.name == name
    ]

    return only(versions)
end

Pkg.activate(; temp = true)
Pkg.Registry.update()

Pkg.develop(Pkg.PackageSpec(; path = PACKAGE_ROOT))
Pkg.add([
    Pkg.PackageSpec(; name = "CIMOptimizer", version = v"0.2.2"),
    Pkg.PackageSpec(; name = "QiskitOpt", version = v"0.7.1"),
    Pkg.PackageSpec(; name = "PySA", version = v"0.4.2"),
    Pkg.PackageSpec(; name = "CondaPkg", version = v"0.2.36"),
])

import CondaPkg

CondaPkg.resolve()

println("DWave version=", _package_version("DWave"))
println("CIMOptimizer version=", _package_version("CIMOptimizer"))
println("QiskitOpt version=", _package_version("QiskitOpt"))
println("PySA version=", _package_version("PySA"))
println("Shared CondaPkg resolve succeeded")
