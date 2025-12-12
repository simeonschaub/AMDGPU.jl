export ROCArray, ROCVector, ROCMatrix, ROCVecOrMat, roc, is_device, is_unified, is_host


## array type

function hasfieldcount(@nospecialize(dt))
    try
        fieldcount(dt)
    catch
        return false
    end
    return true
end

explain_nonisbits(@nospecialize(T), depth = 0) = "  "^depth * "$T is not a bitstype\n"

function explain_eltype(@nospecialize(T), depth = 0; maxdepth = 10)
    depth > maxdepth && return ""

    if T isa Union
        msg = "  "^depth * "$T is a union that's not allocated inline\n"
        for U in Base.uniontypes(T)
            if !Base.allocatedinline(U)
                msg *= explain_eltype(U, depth + 1)
            end
        end
    elseif Base.ismutabletype(T) && Base.datatype_fieldcount(T) != 0
        msg = "  "^depth * "$T is a mutable type\n"
    elseif hasfieldcount(T)
        msg = "  "^depth * "$T is a struct that's not allocated inline\n"
        for U in fieldtypes(T)
            if !Base.allocatedinline(U)
                msg *= explain_nonisbits(U, depth + 1)
            end
        end
    else
        msg = "  "^depth * "$T is not allocated inline\n"
    end
    return msg
end

# ROCArray only supports element types that are allocated inline (`Base.allocatedinline`).
# These come in three forms:
# 1. plain bitstypes (`Int`, `(Float32, Float64)`, plain immutable structs, etc).
#    these are simply stored contiguously in memory.
# 2. structs of unions (`struct Foo; x::Union{Int, Float32}; end`)
#    these are stored with a selector at the end (handled by Julia).
# 3. bitstype unions (`Union{Int, Float32}`, etc)
#    these are stored contiguously and require a selector array (handled by us)
# As well as "mutable singleton" types like `Symbol` that use pointer-identity

function valid_type(@nospecialize(T))
    if Base.allocatedinline(T)
        if hasfieldcount(T)
            return all(valid_type, fieldtypes(T))
        end
        return true
    elseif Base.ismutabletype(T)
        return Base.datatype_fieldcount(T) == 0
    end
    return false
end

@inline function check_eltype(name, T)
    return if !valid_type(T)
        explanation = explain_eltype(T)
        error(
            """
            $name only supports element types that are allocated inline.
            $explanation"""
        )
    end
end

mutable struct ROCArray{T, N, M} <: AbstractGPUArray{T, N}
    data::DataRef{Managed{M}}

    maxsize::Int  # maximum data size; excluding any selector bytes
    offset::Int   # offset of the data in memory, in number of elements

    dims::Dims{N}

    function ROCArray{T, N, M}(::UndefInitializer, dims::Dims{N}) where {T, N, M}
        check_eltype("ROCArray", T)
        maxsize = prod(dims) * aligned_sizeof(T)
        bufsize = if Base.isbitsunion(T)
            # type tag array past the data
            maxsize + prod(dims)
        else
            maxsize
        end

        data = GPUArrays.cached_alloc((ROCArray, device(), M, bufsize)) do
            DataRef(pool_free, pool_alloc(M, bufsize))
        end
        obj = new{T, N, M}(data, maxsize, 0, dims)
        finalizer(unsafe_free!, obj)
        return obj
    end

    function ROCArray{T, N}(
            data::DataRef{Managed{M}}, dims::Dims{N};
            maxsize::Int = prod(dims) * aligned_sizeof(T), offset::Int = 0
        ) where {T, N, M}
        check_eltype("ROCArray", T)
        obj = new{T, N, M}(data, maxsize, offset, dims)
        finalizer(unsafe_free!, obj)
        return obj
    end
end

GPUArrays.storage(a::ROCArray) = a.data


## alias detection

Base.dataids(A::ROCArray) = (UInt(pointer(A)),)

Base.unaliascopy(A::ROCArray) = copy(A)

function Base.mightalias(A::ROCArray, B::ROCArray)
    rA = pointer(A):(pointer(A) + sizeof(A))
    rB = pointer(B):(pointer(B) + sizeof(B))
    return first(rA) <= first(rB) < last(rA) || first(rB) <= first(rA) < last(rB)
end


## convenience constructors

const ROCVector{T} = ROCArray{T, 1}
const ROCMatrix{T} = ROCArray{T, 2}
const ROCVecOrMat{T} = Union{ROCVector{T}, ROCMatrix{T}}

# unspecified memory allocation
const default_memory = let str = Preferences.@load_preference("default_memory", "device")
    if str == "device"
        DeviceMemory
    elseif str == "unified"
        UnifiedMemory
    elseif str == "host"
        HostMemory
    else
        error("unknown default memory type: $default_memory")
    end
end
ROCArray{T, N}(::UndefInitializer, dims::Dims{N}) where {T, N} =
    ROCArray{T, N, default_memory}(undef, dims)

# memory, type and dimensionality specified
ROCArray{T, N, M}(::UndefInitializer, dims::NTuple{N, Integer}) where {T, N, M} =
    ROCArray{T, N, M}(undef, convert(Tuple{Vararg{Int}}, dims))
ROCArray{T, N, M}(::UndefInitializer, dims::Vararg{Integer, N}) where {T, N, M} =
    ROCArray{T, N, M}(undef, convert(Tuple{Vararg{Int}}, dims))

# type and dimensionality specified
ROCArray{T, N}(::UndefInitializer, dims::NTuple{N, Integer}) where {T, N} =
    ROCArray{T, N}(undef, convert(Tuple{Vararg{Int}}, dims))
ROCArray{T, N}(::UndefInitializer, dims::Vararg{Integer, N}) where {T, N} =
    ROCArray{T, N}(undef, convert(Tuple{Vararg{Int}}, dims))

# type but not dimensionality specified
ROCArray{T}(::UndefInitializer, dims::NTuple{N, Integer}) where {T, N} =
    ROCArray{T, N}(undef, convert(Tuple{Vararg{Int}}, dims))
ROCArray{T}(::UndefInitializer, dims::Vararg{Integer, N}) where {T, N} =
    ROCArray{T, N}(undef, convert(Tuple{Vararg{Int}}, dims))

# empty vector constructor
ROCArray{T, 1, M}() where {T, M} = ROCArray{T, 1, M}(undef, 0)
ROCArray{T, 1}() where {T} = ROCArray{T, 1}(undef, 0)

# do-block constructors
for (ctor, tvars) in (
        :ROCArray => (),
        :(ROCArray{T}) => (:T,),
        :(ROCArray{T, N}) => (:T, :N),
        :(ROCArray{T, N, M}) => (:T, :N, :M),
    )
    @eval begin
        function $ctor(f::Function, args...) where {$(tvars...)}
            xs = $ctor(args...)
            return try
                f(xs)
            finally
                unsafe_free!(xs)
            end
        end
    end
end

Base.similar(a::ROCArray{T, N, M}) where {T, N, M} =
    ROCArray{T, N, M}(undef, size(a))
Base.similar(a::ROCArray{T, <:Any, M}, dims::Base.Dims{N}) where {T, N, M} =
    ROCArray{T, N, M}(undef, dims)
Base.similar(a::ROCArray{<:Any, <:Any, M}, ::Type{T}, dims::Base.Dims{N}) where {T, N, M} =
    ROCArray{T, N, M}(undef, dims)

function Base.copy(a::ROCArray{T, N}) where {T, N}
    b = similar(a)
    return @inbounds copyto!(b, a)
end

function Base.deepcopy_internal(x::ROCArray, dict::IdDict)
    haskey(dict, x) && return dict[x]::typeof(x)
    return dict[x] = copy(x)
end


## unsafe_wrap

"""
  # simple case, wrapping a ROCArray around an existing GPU pointer
  unsafe_wrap(ROCArray, ptr::ROCPtr{T}, dims; own=false, device=device())

  # wraps a CPU array object around a unified GPU array
  unsafe_wrap(Array, a::ROCArray)

  # wraps a GPU array object around a CPU array.
  # if your system supports HMM, this is a fast operation.
  # in other cases, it has to use page locking, which can be slow.
  unsafe_wrap(ROCArray, ptr::ptr{T}, dims)
  unsafe_wrap(ROCArray, a::Array)

Wrap a `ROCArray` object around the data at the address given by the HIP-managed pointer
`ptr`. The element type `T` determines the array element type. `dims` is either an integer
(for a 1d array) or a tuple of the array dimensions. `own` optionally specified whether
Julia should take ownership of the memory, calling `hipFree` when the array is no longer
referenced. The `device` argument determines the device where the data is allocated.
"""
unsafe_wrap

# managed pointer to ROCArray
function Base.unsafe_wrap(
        ::Union{Type{ROCArray}, Type{ROCArray{T}}, Type{ROCArray{T, N}}},
        ptr::ROCPtr{T}, dims::NTuple{N, Int};
        own::Bool = false, device::HIPDevice = device()
    ) where {T, N}
    # identify the memory type
    M = try
        typ = memory_type(ptr)
        if is_managed(ptr)
            UnifiedMemory
        elseif typ == hipMemoryTypeDevice
            DeviceMemory
        elseif typ == hipMemoryTypeHost
            HostMemory
        else
            error("Unknown memory type; please file an issue.")
        end
    catch err
        throw(ArgumentError("Could not identify the memory type; are you passing a valid HIP pointer to unsafe_wrap?"))
    end

    return unsafe_wrap(ROCArray{T, N, M}, ptr, dims; own, device)
end
function Base.unsafe_wrap(
        ::Type{ROCArray{T, N, M}},
        ptr::ROCPtr{T}, dims::NTuple{N, Int};
        own::Bool = false, device::HIPDevice = device()
    ) where {T, N, M}
    check_eltype("unsafe_wrap(ROCArray, ...)", T)
    sz = prod(dims) * aligned_sizeof(T)

    # create a memory object
    mem = if M == UnifiedMemory
        UnifiedMemory(device, ptr, sz)
    elseif M == DeviceMemory
        # TODO: can we identify whether this pointer was allocated asynchronously?
        DeviceMemory(device, ptr, sz, false)
    elseif M == HostMemory
        HostMemory(device, host_pointer(ptr), sz)
    else
        throw(ArgumentError("Unknown memory type $M"))
    end

    data = DataRef(own ? pool_free : Returns(nothing), Managed(mem))
    return ROCArray{T, N}(data, dims)
end
# integer size input
function Base.unsafe_wrap(
        ::Union{Type{ROCArray}, Type{ROCArray{T}}, Type{ROCArray{T, 1}}},
        p::ROCPtr{T}, dim::Int;
        own::Bool = false, ctx::HIPDevice = context()
    ) where {T}
    return unsafe_wrap(ROCArray{T, 1}, p, (dim,); own, ctx)
end
function Base.unsafe_wrap(
        ::Type{ROCArray{T, 1, M}}, p::ROCPtr{T}, dim::Int;
        own::Bool = false, ctx::HIPDevice = context()
    ) where {T, M}
    return unsafe_wrap(ROCArray{T, 1, M}, p, (dim,); own, ctx)
end

# managed pointer to Array
function Base.unsafe_wrap(
        ::Union{Type{Array}, Type{Array{T}}, Type{Array{T, N}}},
        p::ROCPtr{T}, dims::NTuple{N, Int};
        own::Bool = false
    ) where {T, N}
    if !is_managed(p) && memory_type(p) != hipMemoryTypeHost
        throw(ArgumentError("Can only create a CPU array object from a unified or host CUDA array"))
    end
    return unsafe_wrap(Array{T, N}, reinterpret(Ptr{T}, p), dims; own)
end
# integer size input
function Base.unsafe_wrap(
        ::Union{Type{Array}, Type{Array{T}}, Type{Array{T, 1}}},
        p::ROCPtr{T}, dim::Int; own::Bool = false
    ) where {T}
    return unsafe_wrap(Array{T, 1}, p, (dim,); own)
end
# array input
function Base.unsafe_wrap(
        ::Union{Type{Array}, Type{Array{T}}, Type{Array{T, N}}},
        a::ROCArray{T, N}
    ) where {T, N}
    p = pointer(a; type = HostMemory)
    return unsafe_wrap(Array, p, size(a))
end

# unmanaged pointer to ROCArray
supports_hmm(dev) = runtime_version() >= v"12.2" &&
    attribute(dev, hipDeviceAttributePageableMemoryAccess) == 1
function Base.unsafe_wrap(
        ::Type{ROCArray{T, N, M}}, p::Ptr{T}, dims::NTuple{N, Int};
        ctx::HIPDevice = context()
    ) where {T, N, M <: AbstractMemory}
    isbitstype(T) || throw(ArgumentError("Can only unsafe_wrap a pointer to a bits type"))
    sz = prod(dims) * aligned_sizeof(T)

    data = if M == UnifiedMemory
        # HMM extends unified memory to include system memory
        supports_hmm(device(ctx)) ||
            throw(ArgumentError("Cannot wrap system memory as unified memory on your system"))
        mem = UnifiedMemory(ctx, reinterpret(ROCPtr{Nothing}, p), sz)
        DataRef(Returns(nothing), Managed(mem))
    elseif M == HostMemory
        # register as device-accessible host memory
        mem = context!(ctx) do
            register(HostMemory, p, sz, hipHostRegisterMapped)
        end
        DataRef(Managed(mem)) do args...
            context!(ctx; skip_destroyed = true) do
                unregister(mem)
            end
        end
    else
        throw(ArgumentError("Cannot wrap system memory as $M"))
    end

    return ROCArray{T, N}(data, dims)
end
function Base.unsafe_wrap(
        ::Union{Type{ROCArray}, Type{ROCArray{T}}, Type{ROCArray{T, N}}},
        p::Ptr{T}, dims::NTuple{N, Int}; ctx::HIPDevice = context()
    ) where {T, N}
    return if supports_hmm(device(ctx))
        Base.unsafe_wrap(ROCArray{T, N, UnifiedMemory}, p, dims; ctx)
    else
        Base.unsafe_wrap(ROCArray{T, N, HostMemory}, p, dims; ctx)
    end
end
# integer size input
Base.unsafe_wrap(
    ::Union{Type{ROCArray}, Type{ROCArray{T}}, Type{ROCArray{T, 1}}},
    p::Ptr{T}, dim::Int
) where {T} =
    unsafe_wrap(ROCArray{T, 1}, p, (dim,))
Base.unsafe_wrap(::Type{ROCArray{T, 1, M}}, p::Ptr{T}, dim::Int) where {T, M} =
    unsafe_wrap(ROCArray{T, 1, M}, p, (dim,))
# array input
Base.unsafe_wrap(
    ::Union{Type{ROCArray}, Type{ROCArray{T}}, Type{ROCArray{T, N}}},
    a::Array{T, N}
) where {T, N} =
    unsafe_wrap(ROCArray{T, N}, pointer(a), size(a))
Base.unsafe_wrap(::Type{ROCArray{T, N, M}}, a::Array{T, N}) where {T, N, M} =
    unsafe_wrap(ROCArray{T, N, M}, pointer(a), size(a))


## array interface

Base.elsize(::Type{<:ROCArray{T}}) where {T} = aligned_sizeof(T)

Base.size(x::ROCArray) = x.dims
Base.sizeof(x::ROCArray) = Base.elsize(x) * length(x)

context(A::ROCArray) = A.data[].mem.dev
device(A::ROCArray) = A.data[].mem.dev

memory_type(x::ROCArray) = memory_type(typeof(x))
memory_type(::Type{<:ROCArray{<:Any, <:Any, M}}) where {M} = @isdefined(M) ? M : Any

is_device(a::ROCArray) = memory_type(a) == DeviceMemory
is_unified(a::ROCArray) = memory_type(a) == UnifiedMemory
is_host(a::ROCArray) = memory_type(a) == HostMemory


## derived types

export DenseROCArray, DenseROCVector, DenseROCMatrix, DenseROCVecOrMat,
    StridedROCArray, StridedROCVector, StridedROCMatrix, StridedROCVecOrMat,
    AnyROCArray, AnyROCVector, AnyROCMatrix, AnyROCVecOrMat

# dense arrays: stored contiguously in memory
#
# all common dense wrappers are currently represented as ROCArray objects.
# this simplifies common use cases, and greatly improves load time.
# AMDGPU.jl 2.0 experimented with using ReshapedArray/ReinterpretArray/SubArray,
# but that proved much too costly. TODO: revisit when we have better Base support.
const DenseROCArray{T, N} = ROCArray{T, N}
const DenseROCVector{T} = DenseROCArray{T, 1}
const DenseROCMatrix{T} = DenseROCArray{T, 2}
const DenseROCVecOrMat{T} = Union{DenseROCVector{T}, DenseROCMatrix{T}}
# XXX: these dummy aliases (DenseROCArray=ROCArray) break alias printing, as
#      `Base.print_without_params` only handles the case of a single alias.

# strided arrays
const StridedSubROCArray{
    T, N, I <: Tuple{
        Vararg{
            Union{
                Base.RangeIndex, Base.ReshapedUnitRange,
                Base.AbstractCartesianIndex,
            },
        },
    },
} =
    SubArray{T, N, <:ROCArray, I}
const StridedROCArray{T, N} = Union{ROCArray{T, N}, StridedSubROCArray{T, N}}
const StridedROCVector{T} = StridedROCArray{T, 1}
const StridedROCMatrix{T} = StridedROCArray{T, 2}
const StridedROCVecOrMat{T} = Union{StridedROCVector{T}, StridedROCMatrix{T}}

"""
    pointer(::ROCArray, [index=1]; [type=DeviceMemory])

Get the native address of a CUDA array object, optionally at a given location `index`.

The `type` argument indicates what kind of pointer to return, either a GPU-accessible
`ROCPtr` when passing `type=DeviceMemory`, or a CPU-accessible `Ptr` when passing
`type=HostMemory`.

!!! note

    The `type` argument indicates what kind of pointer to return, i.e., where the data will
    be accessed from. This is separate from where the data is stored. For example an array
    backed by `HostMemory` may be accessed from both the CPU and GPU, so it is valid to
    pass `type=HostMemory` or `type=DeviceMemory` (but note that accessing `HostMemory` from
    the GPU is typically slow). That also implies it is not valid to pass
    `type=UnifiedMemory`, as this does not indicate where the pointer will be accessed from.
"""
@inline function Base.pointer(x::StridedROCArray{T}, i::Integer = 1; type = DeviceMemory) where {T}
    PT = if type == DeviceMemory
        ROCPtr{T}
    elseif type == HostMemory
        Ptr{T}
    else
        error("unknown memory type")
    end
    return Base.unsafe_convert(PT, x) + Base._memory_offset(x, i)
end

# anything that's (secretly) backed by a ROCArray
const AnyROCArray{T, N} = Union{ROCArray{T, N}, WrappedArray{T, N, ROCArray, ROCArray{T, N}}}
const AnyROCVector{T} = AnyROCArray{T, 1}
const AnyROCMatrix{T} = AnyROCArray{T, 2}
const AnyROCVecOrMat{T} = Union{AnyROCVector{T}, AnyROCMatrix{T}}


## interop with other arrays

@inline function ROCArray{T, N, M}(xs::AbstractArray{<:Any, N}) where {T, N, M}
    A = ROCArray{T, N, M}(undef, size(xs))
    copyto!(A, convert(Array{T}, xs))
    return A
end

@inline ROCArray{T, N}(xs::AbstractArray{<:Any, N}) where {T, N} =
    ROCArray{T, N, default_memory}(xs)

@inline ROCArray{T, N}(xs::ROCArray{<:Any, N, M}) where {T, N, M} =
    ROCArray{T, N, M}(xs)

# underspecified constructors
ROCArray{T}(xs::AbstractArray{S, N}) where {T, N, S} = ROCArray{T, N}(xs)
(::Type{ROCArray{T, N} where {T}})(x::AbstractArray{S, N}) where {S, N} = ROCArray{S, N}(x)
ROCArray(A::AbstractArray{T, N}) where {T, N} = ROCArray{T, N}(A)

# copy xs to match Array behavior
ROCArray{T, N, M}(xs::ROCArray{T, N, M}) where {T, N, M} = copy(xs)
ROCArray{T, N}(xs::ROCArray{T, N, M}) where {T, N, M} = copy(xs)


## conversions

Base.convert(::Type{T}, x::T) where {T <: ROCArray} = x

# defer the conversion to Managed, where we handle memory consistency
# XXX: conversion to Memory or Managed memory by cconvert?
Base.unsafe_convert(typ::Type{Ptr{T}}, x::ROCArray{T}) where {T} =
    convert(typ, x.data[]) + x.offset * Base.elsize(x)
Base.unsafe_convert(typ::Type{ROCPtr{T}}, x::ROCArray{T}) where {T} =
    convert(typ, x.data[]) + x.offset * Base.elsize(x)


## indexing

function Base.getindex(x::ROCArray{<:Any, <:Any, <:Union{HostMemory, UnifiedMemory}}, I::Int)
    @boundscheck checkbounds(x, I)
    return unsafe_load(pointer(x, I; type = HostMemory))
end

function Base.setindex!(x::ROCArray{<:Any, <:Any, <:Union{HostMemory, UnifiedMemory}}, v, I::Int)
    @boundscheck checkbounds(x, I)
    return unsafe_store!(pointer(x, I; type = HostMemory), v)
end


## interop with device arrays

function Base.unsafe_convert(::Type{ROCDeviceArray{T, N, AS.Global}}, a::DenseROCArray{T, N}) where {T, N}
    return ROCDeviceArray{T, N, AS.Global}(
        reinterpret(LLVMPtr{T, AS.Global}, pointer(a)), size(a),
        a.maxsize - a.offset * Base.elsize(a)
    )
end

Adapt.adapt_storage(::Adaptor, x::ROCArray{T,N}) where {T,N} =
    Base.unsafe_convert(ROCDeviceArray{T,N,AS.Global}, x)


## synchronization

synchronize(x::ROCArray) = synchronize(x.data[])

"""
    enable_synchronization!(arr::ROCArray, enable::Bool)

By default `ROCArray`s are implicitly synchronized when they are accessed on different CUDA
devices or streams. This may be unwanted when e.g. using disjoint slices of memory across
different tasks. This function allows to enable or disable this behavior.

!!! warning

    Disabling implicit synchronization affects _all_ `ROCArray`s that are referring to the
    same underlying memory. Unsafe use of this API _will_ result in data corruption.

    This API is only provided as an escape hatch, and should not be used without careful
    consideration. If automatic synchronization is generally problematic for your use case,
    it is recommended to figure out a better model instead and file an issue or pull request.
    For more details see [this discussion](https://github.com/JuliaGPU/AMDGPU.jl/issues/2617).
"""
function enable_synchronization!(arr::ROCArray, enable::Bool = true)
    arr.data[].synchronizing = enable
    return arr
end


## memory copying

if VERSION >= v"1.11.0-DEV.753"
    function typetagdata(a::Array, i = 1)
        ptr_or_offset = Int(a.ref.ptr_or_offset)
        return @ccall(jl_genericmemory_typetagdata(a.ref.mem::Any)::Ptr{UInt8}) + ptr_or_offset + i - 1
    end
else
    typetagdata(a::Array, i = 1) = ccall(:jl_array_typetagdata, Ptr{UInt8}, (Any,), a) + i - 1
end
function typetagdata(a::ROCArray, i = 1; type = DeviceMemory)
    PT = if type == DeviceMemory
        ROCPtr{UInt8}
    elseif type == HostMemory
        Ptr{UInt8}
    else
        error("unknown memory type")
    end
    return convert(PT, a.data[]) + a.maxsize + a.offset + i - 1
end

function Base.copyto!(
        dest::DenseROCArray{T}, doffs::Integer, src::Array{T}, soffs::Integer,
        n::Integer
    ) where {T}
    n == 0 && return dest
    @boundscheck checkbounds(dest, doffs)
    @boundscheck checkbounds(dest, doffs + n - 1)
    @boundscheck checkbounds(src, soffs)
    @boundscheck checkbounds(src, soffs + n - 1)
    unsafe_copyto!(dest, doffs, src, soffs, n)
    return dest
end

Base.copyto!(dest::DenseROCArray{T}, src::Array{T}) where {T} =
    copyto!(dest, 1, src, 1, length(src))

function Base.copyto!(
        dest::Array{T}, doffs::Integer, src::DenseROCArray{T}, soffs::Integer,
        n::Integer
    ) where {T}
    n == 0 && return dest
    @boundscheck checkbounds(dest, doffs)
    @boundscheck checkbounds(dest, doffs + n - 1)
    @boundscheck checkbounds(src, soffs)
    @boundscheck checkbounds(src, soffs + n - 1)
    unsafe_copyto!(dest, doffs, src, soffs, n)
    return dest
end

Base.copyto!(dest::Array{T}, src::DenseROCArray{T}) where {T} =
    copyto!(dest, 1, src, 1, length(src))

function Base.copyto!(
        dest::DenseROCArray{T}, doffs::Integer, src::DenseROCArray{T}, soffs::Integer,
        n::Integer
    ) where {T}
    n == 0 && return dest
    @boundscheck checkbounds(dest, doffs)
    @boundscheck checkbounds(dest, doffs + n - 1)
    @boundscheck checkbounds(src, soffs)
    @boundscheck checkbounds(src, soffs + n - 1)
    unsafe_copyto!(dest, doffs, src, soffs, n)
    return dest
end

Base.copyto!(dest::DenseROCArray{T}, src::DenseROCArray{T}) where {T} =
    copyto!(dest, 1, src, 1, length(src))

# general case: use CUDA APIs

# NOTE: we only switch contexts here to avoid illegal memory accesses.
# our current programming model expects users to manage the active device.

function Base.unsafe_copyto!(
        dest::DenseROCArray{T}, doffs,
        src::Array{T}, soffs, n
    ) where {T}
    device!(device(dest)) do
        # the copy below may block in `libcuda`, so it'd be good to perform a nonblocking
        # synchronization here, but the exact cases are hard to know and detect (e.g., unpinned
        # memory normally blocks, but not for all sizes, and not on all memory architectures).
        GC.@preserve src dest begin
            # semantically, it is not safe for this operation to execute asynchronously, because
            # the Array may be collected before the copy starts executing. However, when using
            # unpinned memory, CUDA first stages a copy to a pinned buffer that will outlive
            # the source array, making this operation safe.
            unsafe_copyto!(pointer(dest, doffs), pointer(src, soffs), n; async = true)
            if Base.isbitsunion(T)
                unsafe_copyto!(typetagdata(dest, doffs), typetagdata(src, soffs), n; async = true)
            end
        end
    end
    return dest
end

function Base.unsafe_copyto!(
        dest::Array{T}, doffs,
        src::DenseROCArray{T}, soffs, n
    ) where {T}
    device!(device(src)) do
        # see comment above; this copy may also block in `libcuda` when dealing with e.g.
        # unpinned memory, but even more likely because we need to wait for the GPU to finish
        # so that the expected data is available. because of that, eagerly perform a nonblocking
        # synchronization first as to maximize the time spent executing Julia code.
        synchronize(src)

        GC.@preserve src dest begin
            unsafe_copyto!(pointer(dest, doffs), pointer(src, soffs), n; async = false)
            if Base.isbitsunion(T)
                unsafe_copyto!(typetagdata(dest, doffs), typetagdata(src, soffs), n; async = false)
            end
        end
    end
    return dest
end

function Base.unsafe_copyto!(
        dest::DenseROCArray{T}, doffs,
        src::DenseROCArray{T}, soffs, n
    ) where {T}
    if device(src) == device(dest) ||
            maybe_enable_peer_access(device(src), device(dest)) == 1
        # use direct device-to-device copy
        device!(device(src)) do
            GC.@preserve src dest begin
                unsafe_copyto!(pointer(dest, doffs), pointer(src, soffs), n; async = true)
                if Base.isbitsunion(T)
                    unsafe_copyto!(typetagdata(dest, doffs), typetagdata(src, soffs), n; async = true)
                end
            end
        end
    else
        # stage through host memory
        tmp = Vector{T}(undef, n)
        unsafe_copyto!(tmp, 1, src, soffs, n)
        unsafe_copyto!(dest, doffs, tmp, 1, n)
    end
    return dest
end

# optimization: memcpy on the CPU for Array <-> unified or host arrays

# NOTE: synchronization is best-effort, since we don't keep track of the
#       dependencies and streams using each array backed by unified memory.

function Base.unsafe_copyto!(
        dest::DenseROCArray{T, <:Any, <:Union{UnifiedMemory, HostMemory}}, doffs,
        src::Array{T}, soffs, n
    ) where {T}
    # maintain stream-ordered semantics: even though the pointer conversion should sync when
    # needed, it's possible that misses captured memory, so ensure copying is always correct.
    synchronize(dest)

    GC.@preserve src dest begin
        ptr = pointer(src, soffs)
        unsafe_copyto!(pointer(dest, doffs; type = HostMemory), ptr, n)
        if Base.isbitsunion(T)
            ptr = typetagdata(src, soffs)
            unsafe_copyto!(typetagdata(dest, doffs; type = HostMemory), ptr, n)
        end
    end
    return dest
end

function Base.unsafe_copyto!(
        dest::Array{T}, doffs,
        src::DenseROCArray{T, <:Any, <:Union{UnifiedMemory, HostMemory}}, soffs, n
    ) where {T}
    # maintain stream-ordered semantics: even though the pointer conversion should sync when
    # needed, it's possible that misses captured memory, so ensure copying is always correct.
    synchronize(src)

    GC.@preserve src dest begin
        ptr = pointer(dest, doffs)
        unsafe_copyto!(ptr, pointer(src, soffs; type = HostMemory), n)
        if Base.isbitsunion(T)
            ptr = typetagdata(dest, doffs)
            unsafe_copyto!(ptr, typetagdata(src, soffs; type = HostMemory), n)
        end
    end

    return dest
end

# optimization: memcpy between host or unified arrays without context switching

function Base.unsafe_copyto!(
        dest::DenseROCArray{T, <:Any, <:Union{UnifiedMemory, HostMemory}}, doffs,
        src::DenseROCArray{T}, soffs, n
    ) where {T}
    device!(device(src)) do
        GC.@preserve src dest begin
            unsafe_copyto!(pointer(dest, doffs), pointer(src, soffs), n; async = true)
            if Base.isbitsunion(T)
                unsafe_copyto!(typetagdata(dest, doffs), typetagdata(src, soffs), n; async = true)
            end
        end
    end
    return dest
end

function Base.unsafe_copyto!(
        dest::DenseROCArray{T}, doffs,
        src::DenseROCArray{T, <:Any, <:Union{UnifiedMemory, HostMemory}}, soffs, n
    ) where {T}
    device!(device(dest)) do
        GC.@preserve src dest begin
            unsafe_copyto!(pointer(dest, doffs), pointer(src, soffs), n; async = true)
            if Base.isbitsunion(T)
                unsafe_copyto!(typetagdata(dest, doffs), typetagdata(src, soffs), n; async = true)
            end
        end
    end
    return dest
end

function Base.unsafe_copyto!(
        dest::DenseROCArray{T, <:Any, <:Union{UnifiedMemory, HostMemory}}, doffs,
        src::DenseROCArray{T, <:Any, <:Union{UnifiedMemory, HostMemory}}, soffs, n
    ) where {T}
    GC.@preserve src dest begin
        unsafe_copyto!(pointer(dest, doffs), pointer(src, soffs), n; async = true)
        if Base.isbitsunion(T)
            unsafe_copyto!(typetagdata(dest, doffs), typetagdata(src, soffs), n; async = true)
        end
    end
    return dest
end


## regular gpu array adaptor

# We don't convert isbits types in `adapt`, since they are already
# considered GPU-compatible.

Adapt.adapt_storage(::Type{ROCArray}, xs::AT) where {AT <: AbstractArray} =
    isbitstype(AT) ? xs : convert(ROCArray, xs)

# if specific type parameters are specified, preserve those
Adapt.adapt_storage(::Type{<:ROCArray{T}}, xs::AT) where {T, AT <: AbstractArray} =
    isbitstype(AT) ? xs : convert(ROCArray{T}, xs)
Adapt.adapt_storage(::Type{<:ROCArray{T, N}}, xs::AT) where {T, N, AT <: AbstractArray} =
    isbitstype(AT) ? xs : convert(ROCArray{T, N}, xs)
Adapt.adapt_storage(::Type{<:ROCArray{T, N, M}}, xs::AT) where {T, N, M, AT <: AbstractArray} =
    isbitstype(AT) ? xs : convert(ROCArray{T, N, M}, xs)


## opinionated gpu array adaptor

# eagerly converts Float64 to Float32, for performance reasons

struct ROCArrayKernelAdaptor{M} end

Adapt.adapt_storage(::ROCArrayKernelAdaptor{M}, xs::AbstractArray{T, N}) where {T, N, M} =
    isbits(xs) ? xs : ROCArray{T, N, M}(xs)

Adapt.adapt_storage(::ROCArrayKernelAdaptor{M}, xs::AbstractArray{T, N}) where {T <: AbstractFloat, N, M} =
    isbits(xs) ? xs : ROCArray{Float32, N, M}(xs)

Adapt.adapt_storage(::ROCArrayKernelAdaptor{M}, xs::AbstractArray{T, N}) where {T <: Complex{<:AbstractFloat}, N, M} =
    isbits(xs) ? xs : ROCArray{ComplexF32, N, M}(xs)

# not for Float16
Adapt.adapt_storage(::ROCArrayKernelAdaptor{M}, xs::AbstractArray{T, N}) where {T <: Union{Float16, BFloat16}, N, M} =
    isbits(xs) ? xs : ROCArray{T, N, M}(xs)

"""
    roc(A; unified=false)

Opinionated GPU array adaptor, which may alter the element type `T` of arrays:
* For `T<:AbstractFloat`, it makes a `ROCArray{Float32}` for performance reasons.
  (Except that `Float16` and `BFloat16` element types are not changed.)
* For `T<:Complex{<:AbstractFloat}` it makes a `ROCArray{ComplexF32}`.
* For other `isbitstype(T)`, it makes a `ROCArray{T}`.

By contrast, `ROCArray(A)` never changes the element type.

Uses Adapt.jl to act inside some wrapper structs.

# Examples

```
julia> roc(ones(3)')
1×3 adjoint(::ROCArray{Float32, 1, AMDGPU.DeviceMemory}) with eltype Float32:
 1.0  1.0  1.0

julia> roc(zeros(1, 3); unified=true)
1×3 ROCArray{Float32, 2, AMDGPU.UnifiedMemory}:
 0.0  0.0  0.0

julia> roc(1:3)
1:3

julia> ROCArray(ones(3)')  # ignores Adjoint, preserves Float64
1×3 ROCArray{Float64, 2, AMDGPU.DeviceMemory}:
 1.0  1.0  1.0

julia> adapt(ROCArray, ones(3)')  # this restores Adjoint wrapper
1×3 adjoint(::ROCArray{Float64, 1, AMDGPU.DeviceMemory}) with eltype Float64:
 1.0  1.0  1.0

julia> ROCArray(1:3)
3-element ROCArray{Int64, 1, AMDGPU.DeviceMemory}:
 1
 2
 3
```
"""
@inline function roc(xs; device::Bool = false, unified::Bool = false, host::Bool = false)
    if device + unified + host > 1
        throw(ArgumentError("Can only specify one of `device`, `unified`, or `host`"))
    end
    memory = if device
        DeviceMemory
    elseif unified
        UnifiedMemory
    elseif host
        HostMemory
    else
        default_memory
    end
    return adapt(ROCArrayKernelAdaptor{memory}(), xs)
end

Base.getindex(::typeof(roc), xs...) = ROCArray([xs...])


## utilities

zeros(T::Type, dims...) = fill!(ROCArray{T}(undef, dims...), zero(T))
ones(T::Type, dims...) = fill!(ROCArray{T}(undef, dims...), one(T))
zeros(dims...) = zeros(Float32, dims...)
ones(dims...) = ones(Float32, dims...)
fill(v, dims...) = fill!(ROCArray{typeof(v)}(undef, dims...), v)
fill(v, dims::Dims) = fill!(ROCArray{typeof(v)}(undef, dims...), v)

# optimized implementation of `fill!` for types that are directly supported by memset
memsettype(T::Type) = T
memsettype(T::Type{<:Signed}) = unsigned(T)
memsettype(T::Type{<:AbstractFloat}) = Base.uinttype(T)
const MemsetCompatTypes = Union{
    UInt8, Int8,
    UInt16, Int16, Float16,
    UInt32, Int32, Float32,
}
function Base.fill!(A::DenseROCArray{T}, x) where {T <: MemsetCompatTypes}
    U = memsettype(T)
    y = reinterpret(U, convert(T, x))
    device!(device(A)) do
        GC.@preserve A memset(convert(ROCPtr{U}, pointer(A)), y, length(A))
    end
    return A
end


## derived arrays

function GPUArrays.derive(::Type{T}, a::ROCArray, dims::Dims{N}, offset::Int) where {T, N}
    offset = (a.offset * Base.elsize(a)) ÷ aligned_sizeof(T) + offset
    return ROCArray{T, N}(copy(a.data), dims; a.maxsize, offset)
end


## views

# pointer conversions
function Base.unsafe_convert(::Type{ROCPtr{T}}, V::SubArray{T, N, P, <:Tuple{Vararg{Base.RangeIndex}}}) where {T, N, P}
    return Base.unsafe_convert(ROCPtr{T}, parent(V)) +
        Base._memory_offset(V.parent, map(first, V.indices)...)
end
function Base.unsafe_convert(::Type{ROCPtr{T}}, V::SubArray{T, N, P, <:Tuple{Vararg{Union{Base.RangeIndex, Base.ReshapedUnitRange}}}}) where {T, N, P}
    return Base.unsafe_convert(ROCPtr{T}, parent(V)) +
        (Base.first_index(V) - 1) * aligned_sizeof(T)
end


## PermutedDimsArray

Base.unsafe_convert(::Type{ROCPtr{T}}, A::PermutedDimsArray) where {T} =
    Base.unsafe_convert(ROCPtr{T}, parent(A))


## resizing

const RESIZE_THRESHOLD = 100 * 1024^2     # 100 MiB
const RESIZE_INCREMENT = 32 * 1024^2     # 32  MiB

"""
  resize!(a::ROCVector, n::Integer)

Resize `a` to contain `n` elements. If `n` is smaller than the current collection length,
the first `n` elements will be retained. If `n` is larger, the new elements are not
guaranteed to be initialized.
"""
function Base.resize!(A::ROCVector{T}, n::Integer) where {T}
    n == length(A) && return A

    # only resize when the new length exceeds the capacity or is much smaller
    cap = A.maxsize ÷ aligned_sizeof(T)
    if n > cap || n < cap ÷ 4
        len = if n < cap
            # shrink to fit
            n
        elseif A.maxsize > RESIZE_THRESHOLD
            # large arrays grown by fixed increments
            max(n, cap + RESIZE_INCREMENT ÷ aligned_sizeof(T))
        else
            # small arrays are doubled in size
            max(n, 2 * length(A))
        end

        # determine the new buffer size
        maxsize = len * aligned_sizeof(T)
        bufsize = if isbitstype(T)
            maxsize
        else
            # type tag array past the data
            maxsize + len
        end

        # allocate new data
        old_data = A.data
        new_data = device!(device(A)) do
            mem = pool_alloc(memory_type(A), bufsize)
            ptr = convert(ROCPtr{T}, mem)
            DataRef(pool_free, mem)
        end

        # replace the data with a new one. this 'unshares' the array.
        # as a result, we can safely support resizing unowned buffers.
        old_pointer = pointer(A)
        old_typetagdata = typetagdata(A)
        A.data = new_data
        A.maxsize = maxsize
        A.offset = 0
        new_pointer = pointer(A)
        new_typetagdata = typetagdata(A)

        # copy existing elements and type tags
        m = min(length(A), n)
        if m > 0
            device!(device(A)) do
                unsafe_copyto!(new_pointer, old_pointer, m; async = true)
                if Base.isbitsunion(T)
                    unsafe_copyto!(new_typetagdata, old_typetagdata, m; async = true)
                end
            end
        end
        unsafe_free!(old_data)
    end

    A.dims = (n,)
    return A
end
