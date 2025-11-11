# Device type and auxiliary functions

export
    HIPDevice, current_device, has_device,
    name, deviceid, uuid, parent_uuid, totalmem, can_access_peer

"""
    HIPDevice(ordinal::Integer)

Get a handle to a compute device.
"""
struct HIPDevice
    handle::HIP.hipDevice_t

    function HIPDevice(ordinal::Integer)
        device_ref = Ref{HIP.hipDevice_t}()
        HIP.hipDeviceGet(device_ref, ordinal)
        new(device_ref[])
    end

    global function current_device()
        device_ref = Ref{HIP.hipDevice_t}()
        res = unchecked_hipCtxGetDevice(device_ref)
        res == ERROR_INVALID_CONTEXT && throw(UndefRefError())
        res != SUCCESS && throw_api_error(res)
        return _HIPDevice(device_ref[])
    end

    # for outer constructors
    global _HIPDevice(handle::HIP.hipDevice_t) = new(handle)
end

"""
    current_device()

Returns the current device.

!!! warning

    This is a low-level API, returning the current device as known to the HIP driver.
    For most users, it is recommended to use the [`device`](@ref) method instead.
"""
current_device()

const DEVICE_CPU = _HIPDevice(HIP.hipDevice_t(-1))
const DEVICE_INVALID = _HIPDevice(HIP.hipDevice_t(-2))

Base.convert(::Type{HIP.hipDevice_t}, dev::HIPDevice) = dev.handle

function Base.show(io::IO, ::MIME"text/plain", dev::HIPDevice)
  print(io, "HIPDevice($(dev.handle)): ")
  if dev == DEVICE_CPU
      print(io, "CPU")
  elseif dev == DEVICE_INVALID
      print(io, "INVALID")
  else
      print(io, "$(name(dev))")
  end
end

"""
    name(dev::HIPDevice)

Returns an identifier string for the device.
"""
function name(dev::HIPDevice)
    buflen = 256
    buf = Vector{Cchar}(undef, buflen)
    HIP.hipDeviceGetName(pointer(buf), buflen, dev)
    buf[end] = 0
    return GC.@preserve buf unsafe_string(pointer(buf))
end

"""
    deviceid(dev::HIPDevice)::Int

Get the ID number of the current device of execution. This is a 0-indexed number,
corresponding to the device ID as known to HIP.
"""
deviceid(dev::HIPDevice) = Int(convert(HIP.hipDevice_t, dev))

function uuid(dev::HIPDevice)
    uuid_ref = Ref{HIP.hipUUID}()
    HIP.hipDeviceGetUuid(uuid_ref, dev)
    Base.UUID(reinterpret(UInt128, reverse([uuid_ref[].bytes...]))[])
end

function parent_uuid(dev::HIPDevice)
    # HIP doesn't have separate parent UUID (no MIG equivalent)
    uuid(dev)
end

"""
    totalmem(dev::HIPDevice)

Returns the total amount of memory (in bytes) on the device.
"""
function totalmem(dev::HIPDevice)
    mem_ref = Ref{Csize_t}()
    HIP.hipDeviceTotalMem(mem_ref, dev)
    return mem_ref[]
end

function can_access_peer(dev::HIPDevice, peer::HIPDevice)
    val_ref = Ref{Cint}()
    HIP.hipDeviceCanAccessPeer(val_ref, dev, peer)
    return val_ref[] == 1
end


## device iteration

export devices, ndevices

struct DeviceIterator end

"""
    devices()

Get an iterator for the compute devices.
"""
devices() = DeviceIterator()

Base.eltype(::DeviceIterator) = HIPDevice

function Base.iterate(iter::DeviceIterator, i=1)
    i >= length(iter) + 1 ? nothing : (HIPDevice(i-1), i+1)
end

Base.length(::DeviceIterator) = ndevices()

Base.IteratorSize(::DeviceIterator) = Base.HasLength()

function Base.show(io::IO, ::MIME"text/plain", iter::DeviceIterator)
    print(io, "AMDGPU.DeviceIterator() for $(length(iter)) devices")
    if !isempty(iter)
        print(io, ":")
        for dev in iter
            print(io, "\n$(deviceid(dev)). $(name(dev))")
        end
    end
end

function ndevices()
    count_ref = Ref{Cint}()
    HIP.hipGetDeviceCount(count_ref)
    return count_ref[]
end


## attributes

export attribute, warpsize, capability, memory_pools_supported, unified_addressing

"""
    attribute(dev::HIPDevice, code)

Returns information about the device.
"""
function attribute(dev::HIPDevice, code::HIP.hipDeviceAttribute_t)
    value_ref = Ref{Cint}()
    HIP.hipDeviceGetAttribute(value_ref, code, dev)
    return value_ref[]
end

@enum_without_prefix HIP.hipDeviceAttribute_t hipDevice

"""
    warpsize(dev::HIPDevice)

Returns the warp size (in threads) of the device.
"""
warpsize(dev::HIPDevice) = attribute(dev, AttributeWarpSize)

"""
    capability(dev::HIPDevice)

Returns the compute capability of the device.
"""
function capability(dev::HIPDevice)
    return VersionNumber(attribute(dev, AttributeComputeCapabilityMajor),
                         attribute(dev, AttributeComputeCapabilityMinor))
end

memory_pools_supported(dev::HIPDevice) = attribute(dev, AttributeMemoryPoolsSupported) == 1
@deprecate has_stream_ordered(dev::HIPDevice) memory_pools_supported(dev)

unified_addressing(dev::HIPDevice) =
    attribute(dev, AttributeManagedMemory) == 1


## p2p attributes

export p2p_attribute

"""
    p2p_attribute(src::HIPDevice, dst::HIPDevice, code)

Returns information about the P2P relationship between a pair of devices.
"""
function p2p_attribute(src::HIPDevice, dst::HIPDevice, code::HIP.hipDeviceP2PAttr)
    value_ref = Ref{Cint}()
    HIP.hipDeviceGetP2PAttribute(value_ref, code, src, dst)
    return value_ref[]
end

@enum_without_prefix HIP.hipDeviceP2PAttr hipDevP2PAttr
