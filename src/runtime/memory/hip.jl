# Device ID => HIPMemoryPool
const MEMORY_POOLS = AMDGPU.LockedObject(Dict{Int64, HIP.HIPMemoryPool}())

function pool_create(dev::HIPDevice)
    mp = MEMORY_POOLS.payload
    did = HIP.device_id(dev)

    pool = get(mp, did, nothing)
    pool ≡ nothing || return pool

    Base.@lock MEMORY_POOLS.lock begin
        get!(mp, HIP.device_id(dev)) do
            max_size::UInt64 = AMDGPU.hard_memory_limit()
            max_size = max_size != typemax(UInt64) ? max_size : UInt64(0)

            pool = HIP.HIPMemoryPool(dev; max_size)
            # Allow pool to use up all device memory.
            soft_limit = AMDGPU.soft_memory_limit()
            HIP.attribute!(pool, HIP.hipMemPoolAttrReleaseThreshold, soft_limit)

            HIP.memory_pool!(dev, pool)
            return pool
        end
    end
end

# ccall integration
Base.unsafe_convert(T, buf::AbstractAMDBuffer) = convert(T, buf)


## device memory

"""
    DeviceMemory

Device memory residing on the GPU.
"""
struct DeviceMemory <: AbstractAMDBuffer
    device::HIPDevice
    ctx::HIPContext
    ptr::Ptr{Cvoid}
    bytesize::Int
    own::Bool
end

function DeviceMemory(bytesize; stream::HIP.HIPStream)
    dev, ctx = stream.device, stream.ctx
    bytesize == 0 && return DeviceMemory(dev, ctx, C_NULL, 0, true)

    AMDGPU.maybe_collect()
    pool = pool_create(dev)

    ptr_ref = Ref{Ptr{Cvoid}}()
    ptr = alloc_or_retry!(isnothing; stream) do
        try
            # Try to allocate.
            HIP.hipMallocAsync(ptr_ref, bytesize, stream)
            # HIP.hipMallocFromPoolAsync(ptr_ref, bytesize, pool, stream)

            ptr = ptr_ref[]
            ptr == C_NULL && throw(HIP.HIPError(HIP.hipErrorOutOfMemory))
            return ptr
        catch err
            # TODO rethrow if not out of memory error
            @debug "hipMallocAsync exception. Requested $(Base.format_bytes(bytesize))." exception=(err, catch_backtrace())
            return nothing
        end
    end
    ptr == C_NULL && throw(HIP.HIPError(HIP.hipErrorOutOfMemory))

    AMDGPU.account!(AMDGPU.memory_stats(dev), bytesize)
    DeviceMemory(dev, ctx, ptr, bytesize, true)
end

function DeviceMemory(ptr::Ptr{Cvoid}, bytesize::Int; own::Bool = false)
    s = AMDGPU.stream()
    DeviceMemory(s.device, s.ctx, ptr, bytesize, own)
end

Base.sizeof(b::DeviceMemory) = UInt64(b.bytesize)

Base.convert(::Type{Ptr{T}}, buf::DeviceMemory) where T = convert(Ptr{T}, buf.ptr)

function view(buf::DeviceMemory, bytesize::Int)
    bytesize > buf.bytesize && throw(BoundsError(buf, bytesize))
    DeviceMemory(buf.device, buf.ctx, buf.ptr + bytesize, buf.bytesize - bytesize, buf.own)
end

function free(buf::DeviceMemory; stream::HIP.HIPStream)
    buf.own || return

    buf.ptr == C_NULL && return
    HIP.hipFreeAsync(buf, stream)
    AMDGPU.account!(AMDGPU.memory_stats(buf.device), -buf.bytesize)
    return
end

function upload!(dst::DeviceMemory, src::Ptr, bytesize::Int; stream::HIP.HIPStream)
    bytesize == 0 && return
    HIP.hipMemcpyHtoDAsync(dst, src, bytesize, stream)
    return
end

function download!(dst::Ptr, src::DeviceMemory, bytesize::Int; stream::HIP.HIPStream)
    bytesize == 0 && return
    HIP.hipMemcpyDtoHAsync(dst, src, bytesize, stream)
    return
end

function transfer!(dst::DeviceMemory, src::DeviceMemory, bytesize::Int; stream::HIP.HIPStream)
    bytesize == 0 && return
    HIP.hipMemcpyDtoDAsync(dst, src, bytesize, stream)
    return
end

## host memory

"""
    HostMemory

Pinned memory residing on the CPU, possibly accessible on the GPU.
"""
struct HostMemory <: AbstractAMDBuffer
    device::HIPDevice
    ctx::HIPContext
    ptr::Ptr{Cvoid}
    dev_ptr::Ptr{Cvoid}
    bytesize::Int
    own::Bool
end

function HostMemory()
    s = AMDGPU.stream()
    HostMemory(s.device, s.ctx, C_NULL, C_NULL, 0, true)
end

function HostMemory(
    bytesize::Integer, flags = 0; stream::HIP.HIPStream = AMDGPU.stream(),
)
    bytesize == 0 && return HostMemory()

    ptr_ref = Ref{Ptr{Cvoid}}()
    HIP.hipHostMalloc(ptr_ref, bytesize, flags)
    ptr = ptr_ref[]
    dev_ptr = get_device_ptr(ptr)
    HostMemory(stream.device, stream.ctx, ptr, dev_ptr, bytesize, true)
end

function HostMemory(
    ptr::Ptr{Cvoid}, sz::Integer;
    stream::HIP.HIPStream = AMDGPU.stream(), own::Bool = false,
)
    pin(ptr, sz)
    dev_ptr = get_device_ptr(ptr)
    HostMemory(stream.device, stream.ctx, ptr, dev_ptr, sz, own)
end

Base.sizeof(b::HostMemory) = UInt64(b.bytesize)

Base.convert(::Type{Ptr{T}}, buf::HostMemory) where T = convert(Ptr{T}, buf.ptr)

@inline device_ptr(buf::HostMemory) = buf.dev_ptr

function view(buf::HostMemory, bytesize::Int)
    bytesize > buf.bytesize && throw(BoundsError(buf, bytesize))
    HostMemory(
        buf.device, buf.ctx,
        buf.ptr + bytesize, buf.dev_ptr + bytesize,
        buf.bytesize - bytesize, buf.own)
end

function free(buf::HostMemory; kwargs...)
    buf.own || return
    buf.ptr == C_NULL && return
    unpin(buf.ptr)
    # TODO
    # call HIP.hipHostFree(buf) if memory was allocated via hipHostMalloc
    # or is unpinning enough?
    return
end

upload!(dst::HostMemory, src::Ptr, sz::Int; stream::HIP.HIPStream) =
    HIP.memcpy(dst, src, sz, HIP.hipMemcpyHostToHost, stream)

upload!(dst::HostMemory, src::DeviceMemory, sz::Int; stream::HIP.HIPStream) =
    HIP.memcpy(dst, src, sz, HIP.hipMemcpyDeviceToHost, stream)

download!(dst::Ptr, src::HostMemory, sz::Int; stream::HIP.HIPStream) =
    HIP.memcpy(dst, src, sz, HIP.hipMemcpyHostToHost, stream)

download!(dst::DeviceMemory, src::HostMemory, sz::Int; stream::HIP.HIPStream) =
    HIP.memcpy(dst, src, sz, HIP.hipMemcpyHostToDevice, stream)

transfer!(dst::HostMemory, src::HostMemory, sz::Int; stream::HIP.HIPStream) =
    HIP.memcpy(dst, src, sz, HIP.hipMemcpyHostToHost, stream)

# download!(::Ptr, ::DeviceMemory)
transfer!(dst::HostMemory, src::DeviceMemory, sz::Int; stream::HIP.HIPStream) =
    HIP.memcpy(dst, src, sz, HIP.hipMemcpyDeviceToHost, stream)

# upload!(::DeviceMemory, ::Ptr)
transfer!(dst::DeviceMemory, src::HostMemory, sz::Int; stream::HIP.HIPStream) =
    HIP.memcpy(dst, src, sz, HIP.hipMemcpyHostToDevice, stream)

## unified memory

"""
    UnifiedMemory

Unified memory buffer that can be accessed from both host and device.
Allocated using `hipMallocManaged` with automatic migration between host and device.

Supports memory advise hints and explicit prefetching for performance optimization.
See: https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management/unified_memory.html
"""
struct UnifiedMemory <: AbstractAMDBuffer
    device::HIPDevice
    ctx::HIPContext
    ptr::Ptr{Cvoid}
    bytesize::Int
    own::Bool
end

function UnifiedMemory(
    bytesize::Integer, flags = HIP.hipMemAttachGlobal;
    stream::HIP.HIPStream = AMDGPU.stream(),
)
    dev, ctx = stream.device, stream.ctx
    bytesize == 0 && return UnifiedMemory(dev, ctx, C_NULL, 0, true)

    AMDGPU.maybe_collect()

    ptr_ref = Ref{Ptr{Cvoid}}()
    HIP.hipMallocManaged(ptr_ref, bytesize, flags)
    ptr = ptr_ref[]
    ptr == C_NULL && throw(HIP.HIPError(HIP.hipErrorOutOfMemory))

    AMDGPU.account!(AMDGPU.memory_stats(dev), bytesize)
    UnifiedMemory(dev, ctx, ptr, bytesize, true)
end

function UnifiedMemory(
    ptr::Ptr{Cvoid}, sz::Integer;
    stream::HIP.HIPStream = AMDGPU.stream(), own::Bool = false,
)
    UnifiedMemory(stream.device, stream.ctx, ptr, sz, own)
end

Base.sizeof(b::UnifiedMemory) = UInt64(b.bytesize)

Base.convert(::Type{Ptr{T}}, buf::UnifiedMemory) where T = convert(Ptr{T}, buf.ptr)

function view(buf::UnifiedMemory, bytesize::Int)
    bytesize > buf.bytesize && throw(BoundsError(buf, bytesize))
    UnifiedMemory(
        buf.device, buf.ctx,
        buf.ptr + bytesize,
        buf.bytesize - bytesize, buf.own)
end

function free(buf::UnifiedMemory; kwargs...)
    buf.own || return
    buf.ptr == C_NULL && return
    HIP.hipFree(buf)
    AMDGPU.account!(AMDGPU.memory_stats(buf.device), -buf.bytesize)
    return
end

# Unified memory can be accessed from both host and device
upload!(dst::UnifiedMemory, src::Ptr, sz::Int; stream::HIP.HIPStream) =
    HIP.memcpy(dst, src, sz, HIP.hipMemcpyHostToHost, stream)

upload!(dst::UnifiedMemory, src::DeviceMemory, sz::Int; stream::HIP.HIPStream) =
    HIP.memcpy(dst, src, sz, HIP.hipMemcpyDeviceToDevice, stream)

upload!(dst::UnifiedMemory, src::HostMemory, sz::Int; stream::HIP.HIPStream) =
    HIP.memcpy(dst, src, sz, HIP.hipMemcpyHostToHost, stream)

download!(dst::Ptr, src::UnifiedMemory, sz::Int; stream::HIP.HIPStream) =
    HIP.memcpy(dst, src, sz, HIP.hipMemcpyHostToHost, stream)

download!(dst::DeviceMemory, src::UnifiedMemory, sz::Int; stream::HIP.HIPStream) =
    HIP.memcpy(dst, src, sz, HIP.hipMemcpyDeviceToDevice, stream)

download!(dst::HostMemory, src::UnifiedMemory, sz::Int; stream::HIP.HIPStream) =
    HIP.memcpy(dst, src, sz, HIP.hipMemcpyHostToHost, stream)

transfer!(dst::UnifiedMemory, src::UnifiedMemory, sz::Int; stream::HIP.HIPStream) =
    HIP.memcpy(dst, src, sz, HIP.hipMemcpyDefault, stream)

transfer!(dst::UnifiedMemory, src::DeviceMemory, sz::Int; stream::HIP.HIPStream) =
    HIP.memcpy(dst, src, sz, HIP.hipMemcpyDeviceToDevice, stream)

transfer!(dst::DeviceMemory, src::UnifiedMemory, sz::Int; stream::HIP.HIPStream) =
    HIP.memcpy(dst, src, sz, HIP.hipMemcpyDeviceToDevice, stream)

transfer!(dst::UnifiedMemory, src::HostMemory, sz::Int; stream::HIP.HIPStream) =
    HIP.memcpy(dst, src, sz, HIP.hipMemcpyHostToHost, stream)

transfer!(dst::HostMemory, src::UnifiedMemory, sz::Int; stream::HIP.HIPStream) =
    HIP.memcpy(dst, src, sz, HIP.hipMemcpyHostToHost, stream)

"""
    prefetch!(buf::UnifiedMemory, device::HIPDevice; stream::HIP.HIPStream)
    prefetch!(buf::UnifiedMemory; stream::HIP.HIPStream)

Prefetch unified memory to the specified device (or the buffer's device).
Explicitly migrates the data to improve performance by reducing page faults.

See: https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/memory_management/unified_memory_reference.html#_CPPv419hipMemPrefetchAsyncPvmi13hipStream_t
"""
function prefetch!(buf::UnifiedMemory, device::HIPDevice; stream::HIP.HIPStream = AMDGPU.stream())
    buf.ptr == C_NULL && return
    HIP.hipMemPrefetchAsync(buf.ptr, buf.bytesize, HIP.device_id(device), stream)
    return
end

function prefetch!(buf::UnifiedMemory; stream::HIP.HIPStream = AMDGPU.stream())
    prefetch!(buf, buf.device; stream)
end

"""
    advise!(buf::UnifiedMemory, advice::HIP.hipMemoryAdvise, device::HIPDevice)
    advise!(buf::UnifiedMemory, advice::HIP.hipMemoryAdvise)

Provide hints to the unified memory system about how the memory will be used.

Available advice flags:
- `hipMemAdviseSetReadMostly`: Data will be mostly read and only occasionally written to
- `hipMemAdviseUnsetReadMostly`: Undo read-mostly advice
- `hipMemAdviseSetPreferredLocation`: Set preferred location for the data
- `hipMemAdviseUnsetPreferredLocation`: Clear preferred location
- `hipMemAdviseSetAccessedBy`: Data will be accessed by specified device
- `hipMemAdviseUnsetAccessedBy`: Clear accessed-by hint
- `hipMemAdviseSetCoarseGrain`: Use coarse-grain coherency (AMD-specific)
- `hipMemAdviseUnsetCoarseGrain`: Use fine-grain coherency (AMD-specific)

See: https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/memory_management/unified_memory_reference.html#_CPPv412hipMemAdvisePvmj8hipMemoryAdvise_t
"""
function advise!(buf::UnifiedMemory, advice::HIP.hipMemoryAdvise, device::HIPDevice)
    buf.ptr == C_NULL && return
    HIP.hipMemAdvise(buf.ptr, buf.bytesize, advice, HIP.device_id(device))
    return
end

function advise!(buf::UnifiedMemory, advice::HIP.hipMemoryAdvise)
    advise!(buf, advice, buf.device)
end

"""
    query_attribute(buf::UnifiedMemory, attribute::HIP.hipMemRangeAttribute)

Query attributes of unified memory range.

Available attributes:
- `hipMemRangeAttributeReadMostly`: Query if the range is read-mostly
- `hipMemRangeAttributePreferredLocation`: Query preferred location
- `hipMemRangeAttributeAccessedBy`: Query which devices can access this range
- `hipMemRangeAttributeLastPrefetchLocation`: Query last prefetch location
- `hipMemRangeAttributeCoherencyMode`: Query coherency mode (AMD-specific)

Returns the attribute value.

See: https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/memory_management/unified_memory_reference.html#_CPPv423hipMemRangeGetAttributePvm20hipMemRangeAttribute_tPvm
"""
function query_attribute(buf::UnifiedMemory, attribute::HIP.hipMemRangeAttribute)
    buf.ptr == C_NULL && error("Cannot query attributes of NULL pointer")

    # Different attributes return different types
    if attribute == HIP.hipMemRangeAttributeReadMostly
        data = Ref{Cint}()
        HIP.hipMemRangeGetAttribute(data, sizeof(Cint), attribute, buf.ptr, buf.bytesize)
        return Bool(data[])
    elseif attribute in (HIP.hipMemRangeAttributePreferredLocation,
                         HIP.hipMemRangeAttributeLastPrefetchLocation)
        data = Ref{Cint}()
        HIP.hipMemRangeGetAttribute(data, sizeof(Cint), attribute, buf.ptr, buf.bytesize)
        return data[]
    elseif attribute == HIP.hipMemRangeAttributeCoherencyMode
        data = Ref{Cuint}()
        HIP.hipMemRangeGetAttribute(data, sizeof(Cuint), attribute, buf.ptr, buf.bytesize)
        return data[]
    else
        # For AccessedBy and other attributes, return raw pointer
        data = Ref{Ptr{Cvoid}}()
        HIP.hipMemRangeGetAttribute(data, sizeof(Ptr{Cvoid}), attribute, buf.ptr, buf.bytesize)
        return data[]
    end
end

## memory utilities

function get_device_ptr(ptr::Ptr{Cvoid})
    ptr == C_NULL && return C_NULL
    ptr_ref = Ref{Ptr{Cvoid}}()
    HIP.hipHostGetDevicePointer(ptr_ref, ptr, 0)
    ptr_ref[]
end

function pin(ptr, sz)
    ptr == C_NULL && error("Cannot pin `NULL` pointer.")

    memtype = attributes(ptr).type
    if memtype == HIP.hipMemoryTypeUnregistered
        HIP.hipHostRegister(ptr, sz, HIP.hipHostRegisterMapped)
    elseif memtype == HIP.hipMemoryTypeHost
        # Already pinned.
    else
        error("Cannot pin pointer with memory type `$memtype`.")
    end
    return
end

function unpin(ptr)
    ptr == C_NULL && error("Cannot unpin `NULL` pointer.")

    memtype = attributes(ptr).type
    if memtype == HIP.hipMemoryTypeUnregistered
        # Already unpinned.
    elseif memtype == HIP.hipMemoryTypeHost
        HIP.hipHostUnregister(ptr)
    else
        error("Cannot unpin pointer with memory type `$memtype`.")
    end
    return
end

function is_pinned(ptr)
    ptr == C_NULL && return false
    data = attributes(ptr)
    return data.type == HIP.hipMemoryTypeHost
end

function attributes(ptr)
    data = Ref{HIP.hipPointerAttribute_t}()
    HIP.hipPointerGetAttributes(data, ptr)
    return data[]
end

"""
Asynchronous 3D array copy.

# Arguments:

- `width::Integer`: Width of 3D memory copy.
- `height::Integer = 1`: Height of 3D memory copy.
- `depth::Integer = 1`: Depth of 3D memory copy.
- `dstPos::ROCDim3 = (1, 1, 1)`:
    Starting position of the destination data for the copy.
- `srcPos::ROCDim3 = (1, 1, 1)`:
    Starting position of the source data for the copy.
- `dstPitch::Integer = 0`: Pitch in bytes of the destination data for the copy.
- `srcPitch::Integer = 0`: Pitch in bytes of the source data for the copy.
- `async::Bool = false`: If `false`, then synchronize `stream` immediately.
- `stream::HIP.HIPStream = AMDGPU.stream()`: Stream on which to perform the copy.
"""
function unsafe_copy3d!(
    dst::Ptr{T}, dstTyp::Type{D},
    src::Ptr{T}, srcTyp::Type{S},
    width::Integer, height::Integer = 1, depth::Integer = 1;
    dstPos::ROCDim = (1, 1, 1), srcPos::ROCDim = (1, 1, 1),
    dstPitch::Integer = 0, dstWidth::Integer = 0, dstHeight::Integer = 0,
    srcPitch::Integer = 0, srcWidth::Integer = 0, srcHeight::Integer = 0,
    async::Bool = false, stream::HIP.HIPStream = AMDGPU.stream(),
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

## backward compatibility aliases

const HIPBuffer = DeviceMemory
const HostBuffer = HostMemory
const HIPUnifiedBuffer = UnifiedMemory
