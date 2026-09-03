module ROCm_Runtime

# Provides the ROCm runtime (HIP, HSA) and the vendor libraries by shipping
# AMD's TheRock distribution tarballs as lazy artifacts, selected by GPU
# architecture and ROCm version through platform augmentation (see `.pkg/`).
# The artifact is a complete ROCm root, laid out like a regular installation.
#
# Setting the "local" preference of this package switches AMDGPU.jl to a local
# ROCm installation instead (see `AMDGPU.set_rocm_version!`); no artifact is
# downloaded or resolved in that case.

using Artifacts, LazyArtifacts, Libdl

# preferences handling and platform selection, shared with Pkg's artifact
# selection hook (`.pkg/select_artifacts.jl`)
include(joinpath(@__DIR__, "..", ".pkg", "platform_augmentation.jl"))

export libamdhip64, libhsa_runtime64, libhiprtc, libamd_comgr
export libMIOpen, libhipblaslt, libhiptensor, librocblas, librocfft, librocrand, librocsolver, librocsparse

global artifact_dir::String = ""
global libamdhip64::String = ""
global libhsa_runtime64::String = ""
global libhiprtc::String = ""
global libamd_comgr::String = ""
global librocblas::String = ""
global librocsparse::String = ""
global librocsolver::String = ""
global librocrand::String = ""
global librocfft::String = ""
global libhipblaslt::String = ""
global libhiptensor::String = ""
global libMIOpen::String = ""

is_available() = !isempty(artifact_dir)

# Resolve the artifact for this host, or "" if no bundle matches (e.g. no
# supported GPU detected, or a local ROCm was requested).
function find_artifact_dir()::String
    local_preference === true && return ""
    dir = try
        @artifact_str("ROCm_Runtime", augment_platform!(HostPlatform()))
    catch err
        @debug "Could not resolve the ROCm_Runtime artifact" exception=(err, catch_backtrace())
        return ""
    end
    # Windows bundles wrap everything in a top-level directory, Linux ones don't.
    entries = readdir(dir)
    if length(entries) == 1 && startswith(only(entries), "therock-dist-")
        dir = joinpath(dir, only(entries))
    end
    return dir
end

# Locate a library in the artifact's library directory, matching both
# unversioned (`libfoo.so`) and versioned (`libfoo.so.N`) names.
function get_library(name::String)::String
    libdir = joinpath(artifact_dir, Sys.iswindows() ? "bin" : "lib")
    isdir(libdir) || return ""
    for file in readdir(libdir)
        if startswith(file, name) && occursin("." * Libdl.dlext, file)
            return joinpath(libdir, file)
        end
    end
    return ""
end

## environment checks

# The bundle's libraries carry a DT_RUNPATH, which the loader searches only after
# LD_LIBRARY_PATH, and they name their ROCm dependencies by soname. A ROCm on
# LD_LIBRARY_PATH (a module, a uenv, AMD's containers) therefore displaces the
# bundle's copies of whatever HIP pulls in transitively (HSA, comgr, ...), and
# the process ends up mixing two releases: comgr then links HIP's blit kernels
# against foreign device libraries, which surfaces as hipErrorOutOfMemory at the
# first stream (https://github.com/ROCm/TheRock/issues/7426). The loader reads
# LD_LIBRARY_PATH once at start-up, so this cannot be undone from inside the
# process; detect it once HIP is loaded and say what to do instead.
#
# LLVM_PATH is a second route to the same mix: comgr roots its clang driver there
# and takes that tree's device libraries.

# The soname-like part of a library file name: up to `.so` and its first version
# component, so that `libhsa-runtime64.so.1.21.0` and `libhsa-runtime64.so.1`
# compare equal, while `libLLVM.so.23.0git` and Julia's `libLLVM.so.18.1jl` do not.
function library_key(name::AbstractString)
    m = match(r"^.+?\.so(?:\.\d+)?", name)
    return m === nothing ? nothing : String(m.match)
end

# Libraries loaded into this process from outside the bundle that the bundle
# also ships, i.e. that shadow it.
function shadowed_libraries()::Vector{String}
    (Sys.islinux() && is_available()) || return String[]
    libdir = joinpath(artifact_dir, "lib")
    bundled = Set{String}()
    for dir in (libdir, joinpath(libdir, "llvm", "lib"), joinpath(libdir, "rocm_sysdeps", "lib"))
        isdir(dir) || continue
        for file in readdir(dir)
            key = library_key(file)
            key === nothing || push!(bundled, key)
        end
    end

    bundle_root = realpath(artifact_dir)
    shadowed = String[]
    for line in eachline("/proc/self/maps")
        fields = split(line; limit = 6)
        length(fields) == 6 || continue
        path = String(strip(fields[6]))
        startswith(path, "/") || continue
        startswith(path, bundle_root) && continue
        key = library_key(basename(path))
        key in bundled && push!(shadowed, path)
    end
    return unique!(shadowed)
end

function warn_environment_conflicts()
    is_available() || return
    # loading HIP is what resolves its dependencies; AMDGPU loads it anyway
    isempty(libamdhip64) || Libdl.dlopen(libamdhip64)

    shadowed = shadowed_libraries()
    llvm_path = get(ENV, "LLVM_PATH", nothing)
    isempty(shadowed) && llvm_path === nothing && return

    msg = "Another ROCm installation in the environment is mixed with the ROCm artifact. Expect failures such as hipErrorOutOfMemory at the first stream."
    if !isempty(shadowed)
        msg *= "\n\nLibraries on LD_LIBRARY_PATH shadow the artifact's:\n" *
               join(("  " * path for path in shadowed), "\n")
    end
    if llvm_path !== nothing
        msg *= "\n\nLLVM_PATH=$llvm_path makes comgr compile against that tree's device libraries instead of the artifact's."
    end
    msg *= "\n\nRemove that ROCm from LD_LIBRARY_PATH and unset LLVM_PATH before starting Julia, or use it instead of the artifact with `AMDGPU.set_rocm_version!(local_rocm=true)`."
    @warn msg
    return
end


function __init__()
    global artifact_dir = find_artifact_dir()
    is_available() || return

    lib_prefix = Sys.islinux() ? "lib" : ""

    global libamdhip64 = get_library(Sys.islinux() ? "libamdhip64" : "amdhip64")
    global libhsa_runtime64 = Sys.islinux() ? get_library("libhsa-runtime64") : ""
    global libhiprtc = get_library(lib_prefix * "hiprtc")
    global libamd_comgr = get_library(lib_prefix * "amd_comgr")

    global librocblas = get_library(lib_prefix * "rocblas")
    global librocsparse = get_library(lib_prefix * "rocsparse")
    global librocsolver = get_library(lib_prefix * "rocsolver")
    global librocrand = get_library(lib_prefix * "rocrand")
    global librocfft = get_library(lib_prefix * "rocfft")
    global libhipblaslt = get_library(lib_prefix * "hipblaslt")
    global libhiptensor = get_library(lib_prefix * "hiptensor")
    global libMIOpen = get_library(lib_prefix * "MIOpen")
end

end
