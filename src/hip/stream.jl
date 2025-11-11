# Stream management

export
    HIPStream, default_stream, legacy_stream, per_thread_stream,
    unique_id, priority, priority_range, synchronize, device_synchronize

"""
    HIPStream(; flags=STREAM_DEFAULT, priority=nothing)

Create a HIP stream.
"""
mutable struct HIPStream
    const handle::HIP.hipStream_t
    Base.@atomic valid::Bool

    const ctx::Union{Nothing,HIPContext}

    function HIPStream(; priority::Union{Nothing,Integer}=nothing)
        handle_ref = Ref{HIP.hipStream_t}()
        if priority === nothing
            HIP.hipStreamCreate(handle_ref)
        else
            priority in priority_range() || throw(ArgumentError("Priority is out of range"))
            HIP.hipStreamCreateWithPriority(handle_ref, flags, priority)
        end

        ctx = HIPContext()
        obj = new(handle_ref[], true, ctx)
        finalizer(unsafe_destroy!, obj)
        return obj
    end

    global default_stream() = new(convert(HIP.hipStream_t, C_NULL), true)

    global legacy_stream() = new(convert(HIP.hipStream_t, C_NULL), true)

    global per_thread_stream() = new(HIP.hipStreamPerThread, true)
end

"""
    default_stream()

Return the default stream.

!!! note

    It is generally better to use `stream()` to get a stream object that's local to the
    current task. That way, operations scheduled in other tasks can overlap.
"""
default_stream()

"""
    legacy_stream()

Return a special object to use use an implicit stream with legacy synchronization behavior.

You can use this stream to perform operations that should block on all streams (with the
exception of streams created with `STREAM_NON_BLOCKING`). This matches the old behavior.
"""
legacy_stream()

"""
    per_thread_stream()

Return a special object to use an implicit stream with per-thread synchronization behavior.
This stream object is normally meant to be used with APIs that do not have per-thread
versions of their APIs (i.e. without a `ptsz` or `ptds` suffix).

!!! note

    It is generally not needed to use this type of stream. With AMDGPU.jl, each task already
    gets its own non-blocking stream, and multithreading in Julia is typically
    accomplished using tasks.
"""
per_thread_stream()

Base.unsafe_convert(::Type{HIP.hipStream_t}, s::HIPStream) = s.handle

Base.:(==)(a::HIPStream, b::HIPStream) = a.handle == b.handle
Base.hash(s::HIPStream, h::UInt) = hash(s.handle, h)

@enum_without_prefix HIP.hipStreamCaptureMode hipStream

function unsafe_destroy!(s::HIPStream)
    @assert s.ctx !== nothing "Cannot destroy unassociated stream"
    context!(s.ctx; skip_destroyed=true) do
        HIP.hipStreamDestroy(s)
    end
    Base.@atomic s.valid = false
end

function Base.show(io::IO, stream::HIPStream)
    print(io, "HIPStream(")
    @printf(io, "%p", stream.handle)
    if stream.ctx !== nothing
        print(io, ", ", stream.ctx)
    end
    print(io, ")")
end

function unique_id(s::HIPStream)
    id_ref = Ref{Culonglong}()
    HIP.hipStreamGetId(s, id_ref)
    return id_ref[]
end

"""
    isvalid(s::HIPStream)

Determines if the stream object is still valid, i.e., if it has not been garbage collected.
This is only useful for use in finalizers, which do not guarantee order of execution (i.e.,
a stream may have been destroyed before an object relying on it has).
"""
function isvalid(s::HIPStream)
    return s.valid
end

"""
    isdone(s::HIPStream)

Return `false` if a stream is busy (has task running or queued)
and `true` if that stream is free.
"""
function isdone(s::HIPStream)
    res = unchecked_hipStreamQuery(s)
    if res == ERROR_NOT_READY
        return false
    elseif res == SUCCESS
        return true
    else
        throw_api_error(res)
    end
end

"""
    synchronize([stream::HIPStream]; blocking::Bool=false)

Wait until `stream` has finished executing, with `stream` defaulting to the stream
associated with the current Julia task.

The `blocking` parameter is provided for compatibility but currently ignored,
as HIP stream synchronization is always blocking at the driver level.

See also: [`device_synchronize`](@ref)
"""
function synchronize(stream::HIPStream=stream(); blocking::Bool=false)
    HIP.hipStreamSynchronize(stream)
    return
end

"""
    priority_range()

Return the valid range of stream priorities as a `StepRange` (with step size  1). The lower
bound of the range denotes the least priority (typically 0), with the upper bound
representing the greatest possible priority (typically -1).
"""
function priority_range()
    least_ref = Ref{Cint}()
    greatest_ref = Ref{Cint}()
    HIP.hipDeviceGetStreamPriorityRange(least_ref, greatest_ref)
    step = least_ref[] < greatest_ref[] ? 1 : -1
    return least_ref[]:Cint(step):greatest_ref[]
end


"""
    priority_range(s::HIPStream)

Return the priority of a stream `s`.
"""
function priority(s::HIPStream)
    priority_ref = Ref{Cint}()
    HIP.hipStreamGetPriority(s, priority_ref)
    return priority_ref[]
end
