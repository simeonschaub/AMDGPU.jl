# definition & memory management of ROCArray, closely following CUDA.jl's CuArray

## eltype validation

function hasfieldcount(@nospecialize(dt))
    try
        fieldcount(dt)
    catch
        return false
    end
    return true
end

explain_nonisbits(@nospecialize(T), depth=0) = "  "^depth * "$T is not a bitstype\n"

function explain_eltype(@nospecialize(T), depth=0; maxdepth=10)
    depth > maxdepth && return ""

    if T isa Union
      msg = "  "^depth * "$T is a union that's not allocated inline\n"
      for U in Base.uniontypes(T)
        if !Base.allocatedinline(U)
          msg *= explain_eltype(U, depth+1)
        end
      end
    elseif Base.ismutabletype(T) && Base.datatype_fieldcount(T) != 0
      msg = "  "^depth * "$T is a mutable type\n"
    elseif hasfieldcount(T)
      msg = "  "^depth * "$T is a struct that's not allocated inline\n"
      for U in fieldtypes(T)
          if !Base.allocatedinline(U)
              msg *= explain_nonisbits(U, depth+1)
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
  if !valid_type(T)
    explanation = explain_eltype(T)
    error("""
      $name only supports element types that are allocated inline.
      $explanation""")
  end
end

## array type

"""
    ROCArray{T,N,B} <: AbstractGPUArray{T,N}

`N`-dimensional dense array of element type `T` stored in GPU-accessible memory
of type `B`. `ROCArray` implements Julia's `AbstractArray` interface, so
broadcasting, reductions, and linear algebra run on the GPU.

The memory type `B` is one of:
- `Mem.HIPBuffer`: regular device memory (the default);
- `Mem.UnifiedBuffer`: unified (managed) memory,
  accessible from both the CPU and the GPU with pages migrating on demand;
- `Mem.HostBuffer`: page-locked host memory that
  is accessible by the GPU.

Copy a host array to the device by wrapping it, or allocate directly:

```julia
ROCArray([1, 2, 3])             # copy a host array to the device
ROCArray{Float32}(undef, 4, 4)  # uninitialized 4×4 device matrix
ROCArray{Float32, 2, AMDGPU.Mem.UnifiedBuffer}(undef, 4, 4)  # unified memory
```

Move data back to the host with `Array(x)`. See also [`roc`](@ref), which
copies to the device while narrowing floating-point types to 32-bit, and the
`AMDGPU.zeros` / `AMDGPU.ones` / `AMDGPU.rand` constructors.
"""
mutable struct ROCArray{T, N, B} <: AbstractGPUArray{T, N}
    data::DataRef{Managed{B}}

    maxsize::Int  # maximum data size; excluding any selector bytes
    offset::Int   # offset of the data in memory, in bytes

    dims::Dims{N}

    function ROCArray{T, N, B}(::UndefInitializer, dims::Dims{N}) where {T, N, B <: Mem.AbstractAMDBuffer}
        check_eltype("ROCArray", T)
        maxsize = prod(dims) * aligned_sizeof(T)
        bufsize = if Base.isbitsunion(T)
            # type tag array past the data
            maxsize + prod(dims)
        else
            maxsize
        end

        data = GPUArrays.cached_alloc((ROCArray, AMDGPU.device(), B, bufsize)) do
            @debug "Allocate `T=$T`, `dims=$dims`: $(Base.format_bytes(bufsize))"
            DataRef(pool_free, pool_alloc(B, bufsize))
        end
        obj = new{T, N, B}(data, maxsize, 0, dims)
        return finalizer(unsafe_free!, obj)
    end

    function ROCArray{T, N}(data::DataRef{Managed{B}}, dims::Dims{N};
        maxsize::Int = prod(dims) * aligned_sizeof(T), offset::Int = 0,
    ) where {T, N, B <: Mem.AbstractAMDBuffer}
        check_eltype("ROCArray", T)
        obj = new{T, N, B}(data, maxsize, offset, dims)
        return finalizer(unsafe_free!, obj)
    end
end

GPUArrays.storage(a::ROCArray) = a.data


## alias detection

Base.dataids(A::ROCArray) = (UInt(pointer(A)),)

Base.unaliascopy(A::ROCArray) = copy(A)

function Base.mightalias(A::ROCArray, B::ROCArray)
    rA = pointer(A):pointer(A) + sizeof(A)
    rB = pointer(B):pointer(B) + sizeof(B)
    return first(rA) <= first(rB) < last(rA) || first(rB) <= first(rA) < last(rB)
end


## convenience constructors

const ROCVector{T} = ROCArray{T,1}
const ROCMatrix{T} = ROCArray{T,2}
const ROCVecOrMat{T} = Union{ROCVector{T},ROCMatrix{T}}

# default to device memory, unless overridden by a preference
const default_memory = let str = @load_preference("default_memory", "device")
    if str == "device"
        Mem.HIPBuffer
    elseif str == "unified"
        Mem.UnifiedBuffer
    elseif str == "host"
        Mem.HostBuffer
    else
        error("unknown default memory type: $str")
    end
end

ROCArray{T,N}(::UndefInitializer, dims::Dims{N}) where {T,N} =
    ROCArray{T,N,default_memory}(undef, dims)

# memory, type and dimensionality specified
ROCArray{T,N,B}(::UndefInitializer, dims::NTuple{N, Integer}) where {T,N,B} =
    ROCArray{T,N,B}(undef, convert(Tuple{Vararg{Int}}, dims))
ROCArray{T,N,B}(::UndefInitializer, dims::Vararg{Integer, N}) where {T,N,B} =
    ROCArray{T,N,B}(undef, convert(Tuple{Vararg{Int}}, dims))

# type and dimensionality specified
ROCArray{T,N}(::UndefInitializer, dims::NTuple{N, Integer}) where {T,N} =
    ROCArray{T,N}(undef, convert(Tuple{Vararg{Int}}, dims))
ROCArray{T,N}(::UndefInitializer, dims::Vararg{Integer, N}) where {T,N} =
    ROCArray{T,N}(undef, convert(Tuple{Vararg{Int}}, dims))

# type but not dimensionality specified
ROCArray{T}(::UndefInitializer, dims::NTuple{N, Integer}) where {T,N} =
    ROCArray{T,N}(undef, convert(Tuple{Vararg{Int}}, dims))
ROCArray{T}(::UndefInitializer, dims::Vararg{Integer, N}) where {T, N} =
    ROCArray{T,N}(undef, convert(Tuple{Vararg{Int}}, dims))

# empty vector constructor
ROCArray{T,1,B}() where {T,B} = ROCArray{T,1,B}(undef, 0)
ROCArray{T,1}() where {T} = ROCArray{T,1}(undef, 0)

# do-block constructors
for (ctor, tvars) in (:ROCArray => (),
                      :(ROCArray{T}) => (:T,),
                      :(ROCArray{T,N}) => (:T, :N),
                      :(ROCArray{T,N,B}) => (:T, :N, :B))
    @eval begin
        function $ctor(f::Function, args...) where {$(tvars...)}
            xs = $ctor(args...)
            try
                f(xs)
            finally
                unsafe_free!(xs)
            end
        end
    end
end

Base.similar(a::ROCArray{T, N, B}) where {T, N, B} =
    ROCArray{T, N, B}(undef, size(a))
Base.similar(::ROCArray{T, <:Any, B}, dims::Base.Dims{N}) where {T, N, B} =
    ROCArray{T, N, B}(undef, dims)
Base.similar(::ROCArray{<:Any, <:Any, B}, ::Type{T}, dims::Base.Dims{N}) where {T, N, B} =
    ROCArray{T, N, B}(undef, dims)

function Base.copy(a::ROCArray)
    b = similar(a)
    @inbounds copyto!(b, a)
end

function Base.deepcopy_internal(x::ROCArray, dict::IdDict)
    haskey(dict, x) && return dict[x]::typeof(x)
    return dict[x] = copy(x)
end


## array interface

Base.elsize(::Type{<:ROCArray{T}}) where {T} = aligned_sizeof(T)
Base.size(x::ROCArray) = x.dims
Base.sizeof(x::ROCArray) = Base.elsize(x) * length(x)

"""
    device(A::ROCArray) -> HIPDevice

Return the device associated with the array `A`.
"""
device(A::ROCArray) = A.data[].mem.device

memory_type(x::ROCArray) = memory_type(typeof(x))
memory_type(::Type{<:ROCArray{<:Any, <:Any, B}}) where B = @isdefined(B) ? B : Any

# backwards compatibility
const buftype = memory_type

"""
    is_device(a::ROCArray) -> Bool

Return `true` if `a` is backed by regular device memory (`Mem.HIPBuffer`).
"""
is_device(a::ROCArray) = memory_type(a) == Mem.HIPBuffer

"""
    is_unified(a::ROCArray) -> Bool

Return `true` if `a` is backed by unified (managed) memory (`Mem.UnifiedBuffer`).
"""
is_unified(a::ROCArray) = memory_type(a) == Mem.UnifiedBuffer

"""
    is_host(a::ROCArray) -> Bool

Return `true` if `a` is backed by pinned host memory (`Mem.HostBuffer`).
"""
is_host(a::ROCArray) = memory_type(a) == Mem.HostBuffer


## derived types

# dense arrays: stored contiguously in memory
const DenseROCArray{T,N} = ROCArray{T,N}
const DenseROCVector{T} = DenseROCArray{T,1}
const DenseROCMatrix{T} = DenseROCArray{T,2}
const DenseROCVecOrMat{T} = Union{DenseROCVector{T}, DenseROCMatrix{T}}

# strided arrays
const StridedSubROCArray{T,N,I<:Tuple{Vararg{Union{
    Base.RangeIndex, Base.ReshapedUnitRange, Base.AbstractCartesianIndex,
}}}} = SubArray{T,N,<:ROCArray,I}
const StridedROCArray{T,N} = Union{ROCArray{T,N}, StridedSubROCArray{T,N}}
const StridedROCVector{T} = StridedROCArray{T,1}
const StridedROCMatrix{T} = StridedROCArray{T,2}
const StridedROCVecOrMat{T} = Union{StridedROCVector{T}, StridedROCMatrix{T}}

"""
    pointer(::ROCArray, [index=1]; [type=Mem.HIPBuffer])

Get the native address of a `ROCArray` object, optionally at a given location
`index`.

The `type` argument indicates what kind of pointer to return: a GPU-accessible
pointer when passing `type=Mem.HIPBuffer` (the default), or a CPU-accessible
pointer when passing `type=Mem.HostBuffer`. The latter is only supported for
arrays backed by host or unified memory.
"""
@inline function Base.pointer(x::ROCArray{T}, i::Integer = 1;
                              type = Mem.HIPBuffer) where T
    if type == Mem.HIPBuffer
        Base.unsafe_convert(Ptr{T}, x) + Base._memory_offset(x, i)
    elseif type == Mem.HostBuffer
        host_pointer(Ptr{T}, x.data[]) + x.offset + Base._memory_offset(x, i)
    else
        error("unknown memory type")
    end
end

# anything that's (secretly) backed by a ROCArray
AnyROCArray{T,N} = Union{ROCArray{T,N}, WrappedArray{T,N,ROCArray,ROCArray{T,N}}}
AnyROCVector{T} = AnyROCArray{T,1}
AnyROCMatrix{T} = AnyROCArray{T,2}
AnyROCVecOrMat{T} = Union{AnyROCVector{T}, AnyROCMatrix{T}}


## interop with other arrays

function ROCArray{T,N,B}(xs::AbstractArray{<:Any,N}) where {T,N,B}
    A = ROCArray{T,N,B}(undef, size(xs))
    copyto!(A, convert(Array{T}, xs))
    return A
end

ROCArray{T,N}(xs::AbstractArray{<:Any,N}) where {T,N} = ROCArray{T,N,default_memory}(xs)
ROCArray{T,N}(xs::ROCArray{<:Any,N,B}) where {T,N,B} = ROCArray{T,N,B}(xs)

# underspecified constructors
ROCArray(A::AbstractArray{T,N}) where {T,N} = ROCArray{T,N}(A)
ROCArray{T}(xs::AbstractArray{S,N}) where {T,N,S} = ROCArray{T,N}(xs)
(::Type{ROCArray{T,N} where T})(x::AbstractArray{S,N}) where {S,N} = ROCArray{S,N}(x)

# copy xs to match Array behavior
ROCArray{T,N,B}(xs::ROCArray{T,N,B}) where {T,N,B} = copy(xs)
ROCArray{T,N}(xs::ROCArray{T,N,B}) where {T,N,B} = copy(xs)


## conversions

Base.convert(::Type{T}, x::T) where T <: ROCArray = x

# defer the conversion to Managed, where we handle memory consistency
Base.unsafe_convert(typ::Type{Ptr{T}}, x::ROCArray{T}) where T =
    convert(typ, x.data[]) + x.offset


## indexing

function Base.getindex(x::ROCArray{<:Any, <:Any, <:Union{Mem.HostBuffer, Mem.UnifiedBuffer}}, I::Int)
    @boundscheck checkbounds(x, I)
    unsafe_load(pointer(x, I; type = Mem.HostBuffer))
end

function Base.setindex!(x::ROCArray{<:Any, <:Any, <:Union{Mem.HostBuffer, Mem.UnifiedBuffer}}, v, I::Int)
    @boundscheck checkbounds(x, I)
    unsafe_store!(pointer(x, I; type = Mem.HostBuffer), v)
end


## interop with device arrays

function Base.convert(
    ::Type{ROCDeviceArray{T, N, AS.Global}}, a::ROCArray{T, N},
) where {T, N}
    buf = convert(Mem.AbstractAMDBuffer, a.data[])
    ptr = convert(Ptr{T}, Mem.device_ptr(buf))
    llvm_ptr = AMDGPU.LLVMPtr{T,AS.Global}(ptr + a.offset)
    ROCDeviceArray{T, N, AS.Global}(a.dims, llvm_ptr)
end

function Adapt.adapt_storage(to::Runtime.Adaptor, x::ROCArray{T,N}) where {T,N}
    managed = x.data[]
    push!(to.managed, managed)
    ptr = convert(Ptr{T}, Mem.device_ptr(managed.mem))
    llvm_ptr = AMDGPU.LLVMPtr{T,AS.Global}(ptr + x.offset)
    return ROCDeviceArray{T, N, AS.Global}(x.dims, llvm_ptr)
end


## synchronization

synchronize(x::ROCArray) = synchronize(x.data[])

"""
    enable_synchronization!(arr::ROCArray, enable::Bool=true)

By default `ROCArray`s are implicitly synchronized when they are accessed on
different streams. This may be unwanted when e.g. using disjoint slices of
memory across different tasks. This function allows to enable or disable this
behavior.

!!! warning

    Disabling implicit synchronization affects _all_ `ROCArray`s that are
    referring to the same underlying memory. Unsafe use of this API _will_
    result in data corruption.
"""
function enable_synchronization!(arr::ROCArray, enable::Bool = true)
    arr.data[].synchronizing = enable
    return arr
end


## memory copying

if VERSION >= v"1.11.0-DEV.753"
function typetagdata(a::Array, i = 1)
    ptr_or_offset = Int(a.ref.ptr_or_offset)
    @ccall(jl_genericmemory_typetagdata(a.ref.mem::Any)::Ptr{UInt8}) + ptr_or_offset + i - 1
end
else
typetagdata(a::Array, i = 1) = ccall(:jl_array_typetagdata, Ptr{UInt8}, (Any,), a) + i - 1
end
function typetagdata(a::ROCArray, i = 1; type = Mem.HIPBuffer)
    ptr = if type == Mem.HIPBuffer
        convert(Ptr{UInt8}, a.data[])
    elseif type == Mem.HostBuffer
        host_pointer(Ptr{UInt8}, a.data[])
    else
        error("unknown memory type")
    end
    # for zero-size element types (e.g. singleton unions), the byte offset
    # is always zero, so the corresponding element offset is also zero
    elem_offset = iszero(Base.elsize(a)) ? 0 : a.offset ÷ Base.elsize(a)
    ptr + a.maxsize + elem_offset + i - 1
end

function Base.copyto!(
    dest::DenseROCArray{T}, d_offset::Integer,
    source::Array{T}, s_offset::Integer, amount::Integer,
) where T
    amount == 0 && return dest
    @boundscheck checkbounds(dest, d_offset)
    @boundscheck checkbounds(dest, d_offset + amount - 1)
    @boundscheck checkbounds(source, s_offset)
    @boundscheck checkbounds(source, s_offset + amount - 1)
    unsafe_copyto!(dest, d_offset, source, s_offset, amount)
    return dest
end

Base.copyto!(dest::DenseROCArray{T}, source::Array{T}) where T =
    copyto!(dest, 1, source, 1, length(source))

function Base.copyto!(
    dest::Array{T}, d_offset::Integer,
    source::DenseROCArray{T}, s_offset::Integer, amount::Integer;
    async::Bool = false,
) where T
    amount == 0 && return dest
    @boundscheck checkbounds(dest, d_offset)
    @boundscheck checkbounds(dest, d_offset + amount - 1)
    @boundscheck checkbounds(source, s_offset)
    @boundscheck checkbounds(source, s_offset + amount - 1)
    unsafe_copyto!(dest, d_offset, source, s_offset, amount; async)
    return dest
end

Base.copyto!(dest::Array{T}, source::DenseROCArray{T}) where T =
    copyto!(dest, 1, source, 1, length(source))

function Base.copyto!(
    dest::DenseROCArray{T}, d_offset::Integer,
    source::DenseROCArray{T}, s_offset::Integer, amount::Integer,
) where T
    amount == 0 && return dest
    @boundscheck checkbounds(dest, d_offset)
    @boundscheck checkbounds(dest, d_offset + amount - 1)
    @boundscheck checkbounds(source, s_offset)
    @boundscheck checkbounds(source, s_offset + amount - 1)
    unsafe_copyto!(dest, d_offset, source, s_offset, amount)
    return dest
end

Base.copyto!(dest::DenseROCArray{T}, source::DenseROCArray{T}) where T =
    copyto!(dest, 1, source, 1, length(source))

# general case: use HIP APIs

function Base.unsafe_copyto!(
    dest::DenseROCArray{T}, doffs, src::Array{T}, soffs, n,
) where T
    GC.@preserve src dest begin
        stm = stream()
        Mem.memcpy!(pointer(dest, doffs), pointer(src, soffs),
            n * aligned_sizeof(T); stream = stm)
        if Base.isbitsunion(T)
            Mem.memcpy!(typetagdata(dest, doffs), typetagdata(src, soffs),
                n; stream = stm)
        end
    end
    return dest
end

function Base.unsafe_copyto!(
    dest::Array{T}, doffs, src::DenseROCArray{T}, soffs, n; async::Bool = false,
) where T
    # eagerly perform a nonblocking synchronization first
    # as to maximize the time spent executing Julia code.
    synchronize(src)

    GC.@preserve src dest begin
        stm = stream()
        Mem.memcpy!(pointer(dest, doffs), pointer(src, soffs),
            n * aligned_sizeof(T); stream = stm)
        if Base.isbitsunion(T)
            Mem.memcpy!(typetagdata(dest, doffs), typetagdata(src, soffs),
                n; stream = stm)
        end
        async || synchronize(stm)
    end
    return dest
end

function Base.unsafe_copyto!(
    dest::DenseROCArray{T}, doffs, src::DenseROCArray{T}, soffs, n,
) where T
    GC.@preserve src dest begin
        stm = stream()
        Mem.memcpy!(pointer(dest, doffs), pointer(src, soffs),
            n * aligned_sizeof(T); stream = stm)
        if Base.isbitsunion(T)
            Mem.memcpy!(typetagdata(dest, doffs), typetagdata(src, soffs),
                n; stream = stm)
        end
    end
    return dest
end

# optimization: memcpy on the CPU for Array <-> unified or host arrays

# NOTE: synchronization is best-effort, since we don't keep track of the
#       dependencies and streams using each array backed by unified memory.

function Base.unsafe_copyto!(
    dest::DenseROCArray{T,<:Any,<:Union{Mem.UnifiedBuffer,Mem.HostBuffer}}, doffs,
    src::Array{T}, soffs, n,
) where T
    # maintain stream-ordered semantics: even though the pointer conversion should sync
    # when needed, it's possible that misses captured memory, so ensure copying is
    # always correct.
    synchronize(dest)

    GC.@preserve src dest begin
        unsafe_copyto!(pointer(dest, doffs; type = Mem.HostBuffer), pointer(src, soffs), n)
        if Base.isbitsunion(T)
            unsafe_copyto!(typetagdata(dest, doffs; type = Mem.HostBuffer),
                typetagdata(src, soffs), n)
        end
    end
    return dest
end

function Base.unsafe_copyto!(
    dest::Array{T}, doffs,
    src::DenseROCArray{T,<:Any,<:Union{Mem.UnifiedBuffer,Mem.HostBuffer}}, soffs, n;
    async::Bool = false,
) where T
    # maintain stream-ordered semantics: even though the pointer conversion should sync
    # when needed, it's possible that misses captured memory, so ensure copying is
    # always correct.
    synchronize(src)

    GC.@preserve src dest begin
        unsafe_copyto!(pointer(dest, doffs), pointer(src, soffs; type = Mem.HostBuffer), n)
        if Base.isbitsunion(T)
            unsafe_copyto!(typetagdata(dest, doffs),
                typetagdata(src, soffs; type = Mem.HostBuffer), n)
        end
    end
    return dest
end


## unsafe_wrap

"""
    # wrap a ROCArray object around the data at the address given by the
    # HIP-managed pointer `ptr`, automatically detecting the memory type
    unsafe_wrap(ROCArray, ptr::Ptr{T}, dims; own=false)

    # wrap a CPU array object around a unified or host GPU array
    unsafe_wrap(Array, a::ROCArray)

The `own` argument optionally specifies whether Julia should take ownership of
the memory, freeing it when the array is no longer referenced.
"""
function Base.unsafe_wrap(
    ::Type{<:ROCArray}, ptr::Ptr{T}, dims::NTuple{N, <:Integer};
    own::Bool = false,
) where {T,N}
    # identify the memory type
    attrs = Mem.attributes(Ptr{Cvoid}(ptr))
    B = if attrs.isManaged == 1 || attrs.type == HIP.hipMemoryTypeManaged
        Mem.UnifiedBuffer
    elseif attrs.type == HIP.hipMemoryTypeUnregistered
        Mem.HostBuffer
    elseif attrs.type == HIP.hipMemoryTypeHost
        Mem.HostBuffer
    elseif attrs.type == HIP.hipMemoryTypeDevice
        Mem.HIPBuffer
    else
        error("Unsupported memory type `$(attrs.type)` for pointer.")
    end
    unsafe_wrap(ROCArray{T, N, B}, ptr, dims; own)
end

function Base.unsafe_wrap(
    ::Type{ROCArray{T, N, B}}, ptr::Ptr{T}, dims::NTuple{N, <:Integer};
    own::Bool = false,
) where {T,N,B}
    check_eltype("unsafe_wrap(ROCArray, ...)", T)
    sz = prod(dims) * aligned_sizeof(T)
    buf = B(Ptr{Cvoid}(ptr), sz; own)
    data = DataRef(own ? pool_free : Returns(nothing), Managed(buf))
    return ROCArray{T, N}(data, convert(Tuple{Vararg{Int}}, dims))
end

Base.unsafe_wrap(::Type{<:ROCArray}, ptr::Ptr, dim::Integer; own::Bool=false) =
    unsafe_wrap(ROCArray, ptr, (dim,); own)

Base.unsafe_wrap(::Type{ROCArray{T}}, ptr::Ptr, dims::NTuple{N, <:Integer}; kwargs...) where {T, N} =
    unsafe_wrap(ROCArray, Base.unsafe_convert(Ptr{T}, ptr), dims; kwargs...)

# host array wrapping a unified or host GPU array
function Base.unsafe_wrap(
    ::Union{Type{Array},Type{Array{T}},Type{Array{T,N}}}, a::ROCArray{T,N},
) where {T,N}
    p = pointer(a; type = Mem.HostBuffer)
    unsafe_wrap(Array, p, size(a))
end


## interop with CPU arrays

# We don't convert isbits types in `adapt`, since they are already
# considered GPU-compatible.

Adapt.adapt_storage(::Type{ROCArray}, xs::AT) where {AT<:AbstractArray} =
    isbitstype(AT) ? xs : convert(ROCArray, xs)

# if specific type parameters are specified, preserve those
Adapt.adapt_storage(::Type{<:ROCArray{T}}, xs::AT) where {T, AT<:AbstractArray} =
    isbitstype(AT) ? xs : convert(ROCArray{T}, xs)
Adapt.adapt_storage(::Type{<:ROCArray{T, N}}, xs::AT) where {T, N, AT<:AbstractArray} =
    isbitstype(AT) ? xs : convert(ROCArray{T, N}, xs)
Adapt.adapt_storage(::Type{<:ROCArray{T, N, B}}, xs::AT) where {T, N, B, AT<:AbstractArray} =
    isbitstype(AT) ? xs : convert(ROCArray{T, N, B}, xs)

Adapt.adapt_storage(::Type{Array}, xs::ROCArray) = convert(Array, xs)


## opinionated gpu array adaptor

# eagerly converts Float64 to Float32, for performance reasons

struct ROCArrayKernelAdaptor{M} end

Adapt.adapt_storage(::ROCArrayKernelAdaptor{M}, xs::AbstractArray{T,N}) where {T,N,M} =
    isbits(xs) ? xs : ROCArray{T,N,M}(xs)

Adapt.adapt_storage(::ROCArrayKernelAdaptor{M}, xs::AbstractArray{T,N}) where {T<:AbstractFloat,N,M} =
    isbits(xs) ? xs : ROCArray{Float32,N,M}(xs)

Adapt.adapt_storage(::ROCArrayKernelAdaptor{M}, xs::AbstractArray{T,N}) where {T<:Complex{<:AbstractFloat},N,M} =
    isbits(xs) ? xs : ROCArray{ComplexF32,N,M}(xs)

# not for Float16
Adapt.adapt_storage(::ROCArrayKernelAdaptor{M}, xs::AbstractArray{T,N}) where {T<:Union{Float16,BFloat16},N,M} =
    isbits(xs) ? xs : ROCArray{T,N,M}(xs)

"""
    roc(x; device=false, unified=false, host=false)

Adapt `x` for the GPU: convert arrays to [`ROCArray`](@ref) while **narrowing
floating-point element types to 32-bit** (`Float64`→`Float32`,
`ComplexF64`→`ComplexF32`; `Float16` and `BFloat16` are left unchanged, other
element types are preserved). Like `Adapt.adapt`, it recurses into custom
structs and converts their array fields.

The keyword arguments select the memory type of the resulting arrays:
`device` for regular device memory, `unified` for unified (managed) memory
that is accessible from both the CPU and the GPU, and `host` for page-locked
host memory. At most one may be set to `true`; when none is, the default
memory type is used.

This mirrors CUDA.jl's `cu`. Reach for it when single precision is preferred
(e.g. for performance); use the [`ROCArray`](@ref) constructor directly to keep
the original element type.

```julia
roc([1.0, 2.0])                # 2-element ROCArray{Float32}
roc(1:3)                       # non-float eltype preserved: ROCArray{Int64}
roc([1.0, 2.0]; unified=true)  # backed by unified memory
```
"""
@inline function roc(xs; device::Bool = false, unified::Bool = false, host::Bool = false)
    if device + unified + host > 1
        throw(ArgumentError("Can only specify one of `device`, `unified`, or `host`"))
    end
    memory = if device
        Mem.HIPBuffer
    elseif unified
        Mem.UnifiedBuffer
    elseif host
        Mem.HostBuffer
    else
        default_memory
    end
    adapt(ROCArrayKernelAdaptor{memory}(), xs)
end

Base.getindex(::typeof(roc), xs...) = ROCArray([xs...])


## utilities

ones(dims...) = ones(Float32, dims...)
ones(T::Type, dims...) = fill!(ROCArray{T}(undef, dims...), one(T))
zeros(dims...) = zeros(Float32, dims...)
zeros(T::Type, dims...) = fill!(ROCArray{T}(undef, dims...), zero(T))
fill(v, dims...) = fill!(ROCArray{typeof(v)}(undef, dims...), v)
fill(v, dims::Dims) = fill!(ROCArray{typeof(v)}(undef, dims...), v)

# optimized implementation of `fill!` for types that are directly supported by memset
memsettype(T::Type) = T
memsettype(T::Type{<:Signed}) = unsigned(T)
memsettype(T::Type{<:AbstractFloat}) = Base.uinttype(T)
const MemsetCompatTypes = Union{UInt8, Int8,
                                UInt16, Int16, Float16,
                                UInt32, Int32, Float32}
function Base.fill!(A::DenseROCArray{T}, x) where T <: MemsetCompatTypes
    U = memsettype(T)
    y = reinterpret(U, convert(T, x))
    GC.@preserve A Mem.memset!(convert(Ptr{U}, pointer(A)), y, length(A); stream=stream())
    A
end


## derived arrays

function GPUArrays.derive(::Type{T}, a::ROCArray, dims::Dims{N}, offset::Int) where {T, N}
    offset = a.offset + offset * aligned_sizeof(T)
    ROCArray{T, N}(copy(a.data), dims; a.maxsize, offset)
end


## resizing

const RESIZE_THRESHOLD = 100 * 1024^2     # 100 MiB
const RESIZE_INCREMENT = 32  * 1024^2     # 32  MiB

"""
    resize!(a::ROCVector, n::Integer)

Resize `a` to contain `n` elements. If `n` is smaller than the current
collection length, the first `n` elements will be retained. If `n` is larger,
the new elements are not guaranteed to be initialized.
"""
function Base.resize!(A::ROCVector{T}, n::Integer) where T
    n == length(A) && return A

    # only resize when the new length exceeds the capacity or is much smaller
    cap = A.maxsize ÷ aligned_sizeof(T)
    if n > cap || n < cap ÷ 4
        len = if n < cap
            # shrink to fit
            n
        elseif A.maxsize > RESIZE_THRESHOLD
            # large arrays grow by fixed increments
            max(n, cap + RESIZE_INCREMENT ÷ aligned_sizeof(T))
        else
            # small arrays are doubled in size
            max(n, 2 * length(A))
        end

        # determine the new buffer size
        maxsize = len * aligned_sizeof(T)
        bufsize = if Base.isbitsunion(T)
            # type tag array past the data
            maxsize + len
        else
            maxsize
        end

        # allocate a new buffer
        old_data = A.data
        new_data = DataRef(pool_free, pool_alloc(memory_type(A), bufsize))

        # replace the data with a new one. this 'unshares' the array.
        # as a result, we can safely support resizing unowned buffers.
        stm = stream()
        m = min(length(A), n)
        old_pointer = pointer(A)
        Base.isbitsunion(T) && (old_typetagdata = typetagdata(A);)
        A.data = new_data
        A.maxsize = maxsize
        A.offset = 0

        # copy existing elements and type tags
        if m > 0
            Mem.memcpy!(pointer(A), old_pointer, m * aligned_sizeof(T); stream=stm)
            if Base.isbitsunion(T)
                Mem.memcpy!(typetagdata(A), old_typetagdata, m; stream=stm)
            end
        end
        unsafe_free!(old_data)
    end

    A.dims = (n,)
    return A
end
