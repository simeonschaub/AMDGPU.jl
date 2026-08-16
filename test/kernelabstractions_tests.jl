using Test
using AMDGPU

import KernelAbstractions
include(joinpath(pkgdir(KernelAbstractions), "test", "testsuite.jl"))

AMDGPU.allowscalar(false)

@testset "kernelabstractions" begin

# TODO fix Printing
skip_tests = ["Printing", "sparse"]
if Sys.iswindows()
    # TODO
    # We do not support hostcalls on Windows yet.
    push!(skip_tests, "Convert")
    # Also launches malloc hostcall for some reason...
    push!(skip_tests, "Private")
end

Testsuite.testsuite(
    ROCBackend, "ROCM", AMDGPU, ROCArray, AMDGPU.ROCDeviceArray;
    skip_tests=Set(skip_tests))

@testset "unified memory" begin
    kab = ROCBackend()
    @test KernelAbstractions.supports_unified(kab)

    x = KernelAbstractions.allocate(kab, Float32, (2, 3); unified=true)
    @test x isa ROCArray{Float32, 2, AMDGPU.Mem.UnifiedBuffer}
    @test is_unified(x)

    y = KernelAbstractions.zeros(kab, Float32, (4,); unified=true)
    @test is_unified(y)
    @test y[2] == 0f0  # CPU-side access without @allowscalar

    z = KernelAbstractions.ones(kab, Float32, (4,); unified=true)
    @test is_unified(z)
    @test Array(z) == ones(Float32, 4)

    d = KernelAbstractions.allocate(kab, Float32, (4,))
    @test AMDGPU.memory_type(d) == AMDGPU.default_memory
end

if Sys.islinux()
    # Disable global malloc hostcall started by conversion tests.
    AMDGPU.synchronize(; stop_hostcalls=true)
end

end
