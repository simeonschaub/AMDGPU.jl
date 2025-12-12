# Stream-ordered memory allocator

export HIPMemoryPool, default_memory_pool, memory_pool, memory_pool!, trim,
       attribute, attribute!, access!

mutable struct HIPMemoryPool
    handle::hipMemPool_t
    device::HIPDevice

    function HIPMemoryPool(dev::HIPDevice;
                          maxSize::Integer=0)
        props = Ref(hipMemPoolProps(
            hipMemAllocationTypePinned,
            hipMemHandleTypeNone,
            hipMemLocation(
                hipMemLocationTypeDevice,
                deviceid(dev)
            ),
            C_NULL,
            Csize_t(maxSize),
            ntuple(i->Cuchar(0), 56)  # reserved
        ))
        handle_ref = Ref{hipMemPool_t}()
        hipMemPoolCreate(handle_ref, props)

        new(handle_ref[], dev)
        # NOTE: we cannot attach a finalizer to this object, as the pool can be active
        #       without any references to it (similar to how contexts work).
    end

    global function default_memory_pool(dev::HIPDevice)
        handle_ref = Ref{hipMemPool_t}()
        hipDeviceGetDefaultMemPool(handle_ref, dev)

        ctx = current_context()
        new(handle_ref[], ctx)
    end

    global function memory_pool(dev::HIPDevice)
        handle_ref = Ref{hipMemPool_t}()
        hipDeviceGetMemPool(handle_ref, dev)

        ctx = current_context()
        new(handle_ref[], ctx)
    end
end

function unsafe_destroy!(pool::HIPMemoryPool)
    context!(pool.ctx; skip_destroyed=true) do
        hipMemPoolDestroy(pool)
    end
end

Base.unsafe_convert(::Type{hipMemPool_t}, pool::HIPMemoryPool) = pool.handle

Base.:(==)(a::HIPMemoryPool, b::HIPMemoryPool) = a.handle == b.handle
Base.hash(pool::HIPMemoryPool, h::UInt) = hash(pool.handle, h)

memory_pool!(dev::HIPDevice, pool::HIPMemoryPool) = hipDeviceSetMemPool(dev, pool)

trim(pool::HIPMemoryPool, bytes_to_keep::Integer=0) = hipMemPoolTrimTo(pool, bytes_to_keep)


## pool attributes

"""
    attribute(X, pool::HIPMemoryPool, attr)

Returns attribute `attr` about `pool`. The type of the returned value depends on the
attribute, and as such must be passed as the `X` parameter.
"""
function attribute(::Type{T}, pool::HIPMemoryPool, attr::hipMemPoolAttr) where T
    value = Ref{T}()
    hipMemPoolGetAttribute(pool, attr, value)
    return value[]
end

"""
    attribute!(pool::HIPMemoryPool, attr, val)

Sets attribute `attr` on a memory pool `pool` to `val`.
"""
function attribute!(pool::HIPMemoryPool, attr::hipMemPoolAttr, value)
    hipMemPoolSetAttribute(pool, attr, Ref(value))
    return
end


## pool access

"""
    access!(pool::HIPMemoryPool, dev::HIPDevice, flags::hipMemAccessFlags)
    access!(pool::HIPMemoryPool, devs::Vector{HIPDevice}, flags::hipMemAccessFlags)

Control the visibility of memory pool `pool` on device `dev` or a list of devices `devs`.
"""
function access!(pool::HIPMemoryPool, devs::Vector{HIPDevice}, flags::hipMemAccessFlags)
    map = Vector{hipMemAccessDesc}(undef, length(devs))
    for (i, dev) in enumerate(devs)
        location = hipMemLocation(hipMemLocationTypeDevice, deviceid(dev))
        access = hipMemAccessDesc(location, flags)
        map[i] = access
    end
    hipMemPoolSetAccess(pool, map, length(map))
end
access!(pool::HIPMemoryPool, dev::HIPDevice, flags::hipMemAccessFlags) =
    access!(pool, [dev], flags)

