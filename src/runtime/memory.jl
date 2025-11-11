# Raw memory management

export attribute, attribute!, memory_type, is_managed


#
# operations on memory
#

# a chunk of memory allocated using the HIP APIs. this memory can reside on the host or on
# the GPU. depending on that, the memory object may be `convert`ed to a Ptr or ROCPtr.

abstract type AbstractMemory end

Base.convert(T::Type{<:Union{Ptr,ROCPtr}}, mem::AbstractMemory) =
    throw(ArgumentError("Illegal conversion of a $(typeof(mem)) to a $T"))

# ccall integration
#
# taking the pointer of a buffer means returning the underlying pointer,
# and not the pointer of the buffer object itself.
Base.unsafe_convert(T::Type{<:Union{Ptr,ROCPtr}}, mem::AbstractMemory) = convert(T, mem)


## device memory

"""
    DeviceMemory

Device memory residing on the GPU.
"""
struct DeviceMemory <: AbstractMemory
    dev::HIPDevice
    ctx::HIPContext
    ptr::ROCPtr{Cvoid}
    bytesize::Int

    async::Bool
end

DeviceMemory() = DeviceMemory(device(), context(), ROC_NULL, 0, false)

Base.pointer(mem::DeviceMemory) = mem.ptr
Base.sizeof(mem::DeviceMemory) = mem.bytesize

Base.show(io::IO, mem::DeviceMemory) =
    @printf(io, "DeviceMemory(%s at %p)", Base.format_bytes(sizeof(mem)), Int(pointer(mem)))

Base.convert(::Type{ROCPtr{T}}, mem::DeviceMemory) where {T} =
    convert(ROCPtr{T}, pointer(mem))

"""
    alloc(DeviceMemory, bytesize::Integer;
          [async=false], [stream::HIPStream], [pool::HIPMemoryPool])

Allocate `bytesize` bytes of memory on the device. This memory is only accessible on the
GPU, and requires explicit calls to `unsafe_copyto!`, which wraps `hipMemcpy`,
for access on the CPU.
"""
function alloc(::Type{DeviceMemory}, bytesize::Integer;
               async::Bool=false,
               stream::Union{Nothing,HIPStream}=nothing,
               pool::Union{Nothing,HIPMemoryPool}=nothing)
    bytesize == 0 && return DeviceMemory()

    ptr_ref = Ref{Ptr{Cvoid}}()
    if async
        stream = @something stream AMDGPU.stream()
        if pool !== nothing
            HIP.hipMallocFromPoolAsync(ptr_ref, bytesize, pool, stream)
        else
            HIP.hipMallocAsync(ptr_ref, bytesize, stream)
        end
    else
        HIP.hipMalloc(ptr_ref, bytesize)
    end

    return DeviceMemory(device(), context(), reinterpret(ROCPtr{Cvoid}, ptr_ref[]), bytesize, async)
end

function free(mem::DeviceMemory; stream::Union{Nothing,HIPStream}=nothing)
    pointer(mem) == ROC_NULL && return

    if mem.async
        stream = @something stream AMDGPU.stream()
        HIP.hipFreeAsync(mem, stream)
    else
        HIP.hipFree(mem)
    end
end


## host memory

"""
    HostMemory

Pinned memory residing on the CPU, possibly accessible on the GPU.
"""
struct HostMemory <: AbstractMemory
    ctx::HIPContext
    ptr::Ptr{Cvoid}
    bytesize::Int
end

HostMemory() = HostMemory(context(), C_NULL, 0)

Base.pointer(mem::HostMemory) = mem.ptr
Base.sizeof(mem::HostMemory) = mem.bytesize

Base.show(io::IO, mem::HostMemory) =
    @printf(io, "HostMemory(%s at %p)", Base.format_bytes(sizeof(mem)), Int(pointer(mem)))

Base.convert(::Type{Ptr{T}}, mem::HostMemory) where {T} =
    convert(Ptr{T}, pointer(mem))

function Base.convert(::Type{ROCPtr{T}}, mem::HostMemory) where {T}
    pointer(mem) == C_NULL && return convert(ROCPtr{T}, ROC_NULL)
    ptr_ref = Ref{Ptr{Cvoid}}()
    HIP.hipHostGetDevicePointer(ptr_ref, pointer(mem), #=flags=# 0)
    convert(ROCPtr{T}, reinterpret(ROCPtr{Cvoid}, ptr_ref[]))
end


const MEMHOSTALLOC_PORTABLE = HIP.hipHostMallocPortable
const MEMHOSTALLOC_DEVICEMAP = HIP.hipHostMallocMapped
const MEMHOSTALLOC_WRITECOMBINED = HIP.hipHostMallocWriteCombined

"""
    alloc(HostMemory, bytesize::Integer, [flags])

Allocate `bytesize` bytes of page-locked memory on the host. This memory is accessible from
the CPU, and makes it possible to perform faster memory copies to the GPU. Furthermore, if
`flags` is set to `MEMHOSTALLOC_DEVICEMAP` the memory is also accessible from the GPU. These
accesses are direct, and go through the PCI bus. If `flags` is set to
`MEMHOSTALLOC_PORTABLE`, the memory is considered mapped by all HIP contexts, not just the
one that created the memory, which is useful if the memory needs to be accessed from
multiple devices. Multiple `flags` can be set at one time using a bytewise `OR`:

    flags = MEMHOSTALLOC_PORTABLE | MEMHOSTALLOC_DEVICEMAP

"""
function alloc(::Type{HostMemory}, bytesize::Integer, flags=0)
    bytesize == 0 && return HostMemory()

    ptr_ref = Ref{Ptr{Cvoid}}()
    HIP.hipHostMalloc(ptr_ref, bytesize, flags)

    return HostMemory(context(), ptr_ref[], bytesize)
end


const MEMHOSTREGISTER_PORTABLE = HIP.hipHostRegisterPortable
const MEMHOSTREGISTER_DEVICEMAP = HIP.hipHostRegisterMapped
const MEMHOSTREGISTER_IOMEMORY = HIP.hipHostRegisterIoMemory

"""
    register(HostMemory, ptr::Ptr, bytesize::Integer, [flags])

Page-lock the host memory pointed to by `ptr`. Subsequent transfers to and from devices will
be faster, and can be executed asynchronously. If the `MEMHOSTREGISTER_DEVICEMAP` flag is
specified, the buffer will also be accessible directly from the GPU. These accesses are
direct, and go through the PCI bus. If the `MEMHOSTREGISTER_PORTABLE` flag is specified, any
HIP context can access the memory.
"""
function register(::Type{HostMemory}, ptr::Ptr, bytesize::Integer, flags=0)
    bytesize == 0 && throw(ArgumentError("Cannot register an empty range of memory."))

    HIP.hipHostRegister(ptr, bytesize, flags)

    return HostMemory(context(), ptr, bytesize)
end

"""
    unregister(::HostMemory)

Unregisters a memory range that was registered with [`register`](@ref).
"""
function unregister(mem::HostMemory)
    HIP.hipHostUnregister(mem)
end


function free(mem::HostMemory)
    if pointer(mem) != C_NULL
        HIP.hipHostFree(mem)
    end
end


## unified memory

"""
    UnifiedMemory

Unified memory that is accessible on both the CPU and GPU.
"""
struct UnifiedMemory <: AbstractMemory
    ctx::HIPContext
    ptr::ROCPtr{Cvoid}
    bytesize::Int
end

UnifiedMemory() = UnifiedMemory(context(), ROC_NULL, 0)

Base.pointer(mem::UnifiedMemory) = mem.ptr
Base.sizeof(mem::UnifiedMemory) = mem.bytesize

Base.show(io::IO, mem::UnifiedMemory) =
    @printf(io, "UnifiedMemory(%s at %p)", Base.format_bytes(sizeof(mem)), Int(pointer(mem)))

Base.convert(::Type{Ptr{T}}, mem::UnifiedMemory) where {T} =
    convert(Ptr{T}, reinterpret(Ptr{Cvoid}, pointer(mem)))

Base.convert(::Type{ROCPtr{T}}, mem::UnifiedMemory) where {T} =
    convert(ROCPtr{T}, pointer(mem))

"""
    alloc(UnifiedMemory, bytesize::Integer, [flags])

Allocate `bytesize` bytes of unified memory. This memory is accessible from both the CPU and
GPU, with the HIP runtime automatically migrating upon first access.
"""
function alloc(::Type{UnifiedMemory}, bytesize::Integer,
              flags::Integer=HIP.hipMemAttachGlobal)
    bytesize == 0 && return UnifiedMemory()

    ptr_ref = Ref{Ptr{Cvoid}}()
    HIP.hipMallocManaged(ptr_ref, bytesize, flags)

    return UnifiedMemory(context(), reinterpret(ROCPtr{Cvoid}, ptr_ref[]), bytesize)
end


function free(mem::UnifiedMemory)
    if pointer(mem) != ROC_NULL
        HIP.hipFree(mem)
    end
end


"""
    prefetch(::UnifiedMemory, [bytes::Integer]; [device::HIPDevice], [stream::HIPStream])

Prefetches memory to the specified destination device.
"""
function prefetch(mem::UnifiedMemory, bytes::Integer=sizeof(mem);
                  device::HIPDevice=device(), stream::HIPStream=stream())
    bytes > sizeof(mem) && throw(BoundsError(mem, bytes))
    HIP.hipMemPrefetchAsync(mem, bytes, HIP.device_id(device), stream)
end


"""
    advise(::UnifiedMemory, advice::HIP.hipMemoryAdvise, [bytes::Integer]; [device::HIPDevice])

Advise about the usage of a given memory range.
"""
function advise(mem::UnifiedMemory, advice::HIP.hipMemoryAdvise, bytes::Integer=sizeof(mem);
                device::HIPDevice=device())
    bytes > sizeof(mem) && throw(BoundsError(mem, bytes))
    HIP.hipMemAdvise(mem, bytes, advice, HIP.device_id(device))
end



#
# operations on pointers
#

## initialization

"""
    memset(mem::ROCPtr, value::Union{UInt8,UInt16,UInt32}, len::Integer; [stream::HIPStream])

Initialize device memory by copying `val` for `len` times.
"""
memset

for T in [UInt8, UInt16, UInt32]
    bits = 8*sizeof(T)
    fn = Symbol("hipMemsetD$(bits)Async")
    @eval function memset(ptr::ROCPtr{$T}, value::$T, len::Integer; stream::HIPStream=stream())
        $(getproperty(HIP, fn))(ptr, value, len, stream)
        return
    end
end


## copy operations

# XXX: also provide low-level memcpy?

for (fn, srcPtrTy, dstPtrTy) in (("hipMemcpyDtoHAsync", :ROCPtr, :Ptr),
                                 ("hipMemcpyHtoDAsync", :Ptr,   :ROCPtr),
                                 )
    @eval function Base.unsafe_copyto!(dst::$dstPtrTy{T}, src::$srcPtrTy{T}, N::Integer;
                                       stream::HIPStream=stream(),
                                       async::Bool=false) where T
        $(getproperty(HIP, Symbol(fn)))(dst, src, N*aligned_sizeof(T), stream)
        async || synchronize(stream)
        return dst
    end
end

function Base.unsafe_copyto!(dst::ROCPtr{T}, src::ROCPtr{T}, N::Integer;
                             stream::HIPStream=stream(),
                             async::Bool=false) where T
    # HIP doesn't have separate peer copy, just use regular DtoD
    HIP.hipMemcpyDtoDAsync(dst, src, N*aligned_sizeof(T), stream)
    async || synchronize(stream)
    return dst
end

"""
    unsafe_copy3d!(dst, dstTyp, src, srcTyp, width, height=1, depth=1;
                   dstPos=(1,1,1), dstPitch=0, dstWidth=0, dstHeight=0,
                   srcPos=(1,1,1), srcPitch=0, srcWidth=0, srcHeight=0,
                   async=false, stream=nothing)

Perform a 3D memory copy between pointers `src` and `dst`, at respectively position `srcPos`
and `dstPos` (1-indexed). Both pitch, width and height can be specified for both the source
and destination; consult the HIP documentation for more details. This call is executed
asynchronously if `async` is set, otherwise `stream` is synchronized.
"""
function unsafe_copy3d!(
    dst::Ptr{T}, dstTyp::Type{D},
    src::Ptr{T}, srcTyp::Type{S},
    width::Integer, height::Integer = 1, depth::Integer = 1;
    dstPos::ROCDim = (1, 1, 1), srcPos::ROCDim = (1, 1, 1),
    dstPitch::Integer = 0, dstWidth::Integer = 0, dstHeight::Integer = 0,
    srcPitch::Integer = 0, srcWidth::Integer = 0, srcHeight::Integer = 0,
    async::Bool = false, stream::HIPStream = AMDGPU.stream(),
) where {T, D, S}
    (width == 0 || height == 0 || depth == 0) && return dst

    srcPos, dstPos = ROCDim3(srcPos), ROCDim3(dstPos)
    srcPos = HIP.hipPos((srcPos[1] - 1) * sizeof(T), srcPos[2] - 1, srcPos[3] - 1)
    dstPos = HIP.hipPos((dstPos[1] - 1) * sizeof(T), dstPos[2] - 1, dstPos[3] - 1)

    extent = HIP.hipExtent(width * sizeof(T), height, depth)
    kind = if D <: DeviceMemory && S <: DeviceMemory
        HIP.hipMemcpyDeviceToDevice
    elseif D <: DeviceMemory && S <: HostMemory
        HIP.hipMemcpyHostToDevice
    elseif D <: HostMemory && S <: DeviceMemory
        HIP.hipMemcpyDeviceToHost
    elseif D <: HostMemory && S <: HostMemory
        HIP.hipMemcpyHostToHost
    end

    srcPtr = HIP.hipPitchedPtr(src, srcPitch, srcWidth, srcHeight)
    dstPtr = HIP.hipPitchedPtr(dst, dstPitch, dstWidth, dstHeight)
    params = Ref(HIP.hipMemcpy3DParms(
        C_NULL, srcPos, srcPtr,
        C_NULL, dstPos, dstPtr, extent, kind))

    HIP.hipMemcpy3DAsync(params, stream)
    async || AMDGPU.synchronize(stream)
    return dst
end



#
# auxiliary functionality
#

# given object, find base allocation
# pin that, or increase refcount
# finalizer, drop refcount, free if 0

## memory pinning

const __pin_lock = ReentrantLock()

struct PinnedObject
    ref::WeakRef
    size::Int  # memory size in bytes
end

# - IdDict does not free the memory
# - WeakRef dict does not unique the key by objectid
const __pinned_objects = Dict{Tuple{HIPContext,Ptr{Cvoid}}, PinnedObject}()

function pin(a::AbstractArray)
    ctx = context()
    ptr = pointer(a)

    Base.@lock __pin_lock begin
        # only pin an object once per context
        key = (ctx, convert(Ptr{Nothing}, ptr))
        if haskey(__pinned_objects, key) && __pinned_objects[key].ref.value !== nothing
            if sizeof(a) == __pinned_objects[key].size
                return nothing
            else
                # if the object size has changed, unpin it first; it will be re-pinned with the new size
                __unpin(ptr, ctx)
            end
        end
        __pinned_objects[key] = PinnedObject(WeakRef(a), sizeof(a))
    end

     __pin(ptr, sizeof(a))
    finalizer(a) do _
        __unpin(ptr, ctx)
    end

    a
end

function pin(ref::Base.RefValue{T}) where T
    ctx = context()
    ptr = Base.unsafe_convert(Ptr{T}, ref)

    __pin(ptr, aligned_sizeof(T))
    finalizer(ref) do _
        __unpin(ptr, ctx)
    end

    ref
end

# derived arrays should always pin the parent memory range, because we may end up copying
# from or to that parent range (containing the derived range), and partially-pinned ranges
# are not supported:
#
# > Memory regions requested must be either entirely registered with HIP, or in the case
# > of host pageable transfers, not registered at all. Memory regions spanning over
# > allocations that are both registered and not registered with HIP are not supported and
# > will return an error.
__pin(a::Union{SubArray, Base.ReinterpretArray, Base.ReshapedArray}) = __pin(parent(a))

# refcount the pinning per context, since we can only pin a memory range once
const __pinned_memory = Dict{Tuple{HIPContext,Ptr{Cvoid}}, HostMemory}()
const __pin_count = Dict{Tuple{HIPContext,Ptr{Cvoid}}, Int}()
function __pin(ptr::Ptr, sz::Int)
    ctx = context()
    key = (ctx, convert(Ptr{Nothing}, ptr))

    Base.@lock __pin_lock begin
        pin_count = if haskey(__pin_count, key)
            __pin_count[key] += 1
        else
            __pin_count[key] = 1
        end

        if pin_count == 1
            mem = register(HostMemory, ptr, sz)
            __pinned_memory[key] = mem
        elseif Base.JLOptions().debug_level >= 2
            # make sure we're pinning the exact same range
            @assert haskey(__pinned_memory, key) "Cannot find memory for $ptr with pin count $pin_count."
            mem = __pinned_memory[key]
            @assert sz == sizeof(mem) "Mismatch between pin request of $ptr: $sz vs. $(sizeof(mem))."
        end
    end

    return
end
function __unpin(ptr::Ptr, ctx::HIPContext)
    key = (ctx, convert(Ptr{Nothing}, ptr))

    Base.@lock __pin_lock begin
        @assert haskey(__pin_count, key) "Cannot unpin unmanaged pointer $ptr."
        pin_count = __pin_count[key] -= 1

        if pin_count == 0
            mem = @inbounds __pinned_memory[key]
            context!(ctx; skip_destroyed=true) do
                unregister(mem)
            end
            delete!(__pinned_memory, key)
        end
    end

    return
end
function __pinned(ptr::Ptr, ctx::HIPContext)
    key = (ctx, convert(Ptr{Nothing}, ptr))
    Base.@lock __pin_lock begin
        haskey(__pin_count, key)
    end
end


## pointer attributes

# TODO: iterable struct

"""
    attribute(X, ptr::Union{Ptr,ROCPtr}, attr)

Returns attribute `attr` about pointer `ptr`. The type of the returned value depends on the
attribute, and as such must be passed as the `X` parameter.
"""
function attribute(X::Type, ptr::Union{Ptr{T},ROCPtr{T}}, attr::HIP.hipPointerAttribute_t) where {T}
    data = Ref{HIP.hipPointerAttribute_t}()
    HIP.hipPointerGetAttributes(data, reinterpret(Ptr{Cvoid}, ptr))
    # Return the requested field from the attribute struct
    return getfield(data[], attr)
end

# some common attributes

"""
    memory_type(x)

Identify the memory type of a pointer.
"""
function memory_type(x::Union{Ptr,ROCPtr})
    data = Ref{HIP.hipPointerAttribute_t}()
    HIP.hipPointerGetAttributes(data, reinterpret(Ptr{Cvoid}, x))
    return data[].type
end

is_managed(x::Union{Ptr,ROCPtr}) = memory_type(x) == HIP.hipMemoryTypeUnified

function is_pinned(ptr::Ptr)
    # Check if memory is pinned (registered with HIP)
    data = Ref{HIP.hipPointerAttribute_t}()
    res = HIP.hipPointerGetAttributes(data, ptr)
    if res == HIP.hipSuccess
        return data[].type == HIP.hipMemoryTypeHost
    else
        return false
    end
end



#
# other
#

## memory info

function memory_info()
    free_ref = Ref{Csize_t}()
    total_ref = Ref{Csize_t}()
    HIP.hipMemGetInfo(free_ref, total_ref)
    return convert(Int, free_ref[]), convert(Int, total_ref[])
end

"""
    free_memory()

Returns the free amount of memory (in bytes), available for allocation by the HIP context.
"""
free_memory() = Int(memory_info()[1])

"""
    total_memory()

Returns the total amount of memory (in bytes), available for allocation by the HIP context.
"""
total_memory() = Int(memory_info()[2])
