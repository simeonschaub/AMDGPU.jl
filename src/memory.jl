"""
    info()

Returns a tuple of two integers, indicating respectively the free and total
amount of memory (in bytes) available for allocation on the device.
"""
function info()
    free_ref = Ref{Csize_t}()
    total_ref = Ref{Csize_t}()
    HIP.hipMemGetInfo(free_ref, total_ref)
    return convert(Int, free_ref[]), convert(Int, total_ref[])
end

"""
    free()

Returns the free amount of memory (in bytes),
available for allocation on the device.
"""
free() = info()[1]

"""
    total()

Returns the total amount of memory (in bytes),
available for allocation on the device.
"""
total() = info()[2]

"""
    used()

Returns the used amount of memory (in bytes), allocated on the device.
"""
used() = total() - free()

function parse_memory_limit(limit_str::String)::UInt64
    limit_str == "none" && return typemax(UInt64)

    units = ("%", "MiB", "GiB")

    value, unit = split(limit_str) # TODO check length 2 before split
    unit in units || throw(ArgumentError("""
    Memory limit must be specified in `$units` units, but `$unit` was given.
    """))

    total_memory = total()
    limit = if unit == "%"
        v = parse(Int, value)
        0 < v ≤ 100 || throw(ArgumentError("""
        Invalid percentage value for memory limit `$v`.
        Must be in (0, 100] range or 'none'.
        """))
        floor(UInt64, total_memory * (v / 100))
    else
        scale = unit == "MiB" ? (1024^2) : (1024^3)
        parse(UInt64, value) * scale
    end

    limit > total_memory && throw(ArgumentError("""
    Memory limit `$(Base.format_bytes(limit))` is bigger than the actual memory `$(Base.format_bytes(total_memory))`.
    Set to `none` to disable memory limit.
    """))
    return limit
end

"""
Set a hard limit for total GPU memory allocations.
"""
hard_memory_limit!(limit::String) =
    @set_preferences!("hard_memory_limit" => limit)

soft_memory_limit!(limit::String) =
    @set_preferences!("soft_memory_limit" => limit)

const HARD_MEMORY_LIMIT = Ref{Union{Nothing, UInt64}}(nothing)
function hard_memory_limit()
    hard_limit = HARD_MEMORY_LIMIT[]
    hard_limit ≢ nothing && return hard_limit

    hard_limit = parse_memory_limit(
        @load_preference("hard_memory_limit", "none"))

    @debug "Setting hard memory limit: $(Base.format_bytes(hard_limit))"
    HARD_MEMORY_LIMIT[] = hard_limit
end

const SOFT_MEMORY_LIMIT = Ref{Union{Nothing, UInt64}}(nothing)
function soft_memory_limit()
    soft_limit = SOFT_MEMORY_LIMIT[]
    soft_limit ≢ nothing && return soft_limit

    soft_limit = parse_memory_limit(
        @load_preference("soft_memory_limit", "none"))

    @debug "Setting soft memory limit: $(Base.format_bytes(soft_limit))"
    SOFT_MEMORY_LIMIT[] = soft_limit
end


## allocation statistics

mutable struct AllocStats
    Base.@atomic alloc_count::Int
    Base.@atomic alloc_bytes::Int

    Base.@atomic free_count::Int
    Base.@atomic free_bytes::Int

    Base.@atomic total_time::Float64
end

AllocStats() = AllocStats(0, 0, 0, 0, 0.0)

Base.copy(s::AllocStats) =
    AllocStats(s.alloc_count, s.alloc_bytes,
               s.free_count, s.free_bytes, s.total_time)

Base.:(-)(a::AllocStats, b::AllocStats) = (;
    alloc_count = a.alloc_count - b.alloc_count,
    alloc_bytes = a.alloc_bytes - b.alloc_bytes,
    free_count  = a.free_count  - b.free_count,
    free_bytes  = a.free_bytes  - b.free_bytes,
    total_time  = a.total_time  - b.total_time)

const alloc_stats = AllocStats()


## memory accounting

mutable struct MemoryStats
    # Maximum size of the heap.
    # Estimated during `maybe_collect` stage.
    Base.@atomic size::Int
    Base.@atomic last_updated::Float64

    # Live bytes.
    Base.@atomic live::Int

    # Last `maybe_collect` update.
    Base.@atomic last_time::Float64
    # Amount of time last GC call took.
    Base.@atomic last_gc_time::Float64
    # Amount of freed memory during last `maybe_collect`.
    Base.@atomic last_freed::Int
end

MemoryStats() = MemoryStats(0, 0.0, 0, 0.0, 0.0, 0)

const MEMORY_STATS = AMDGPU.LockedObject(Dict{Int, MemoryStats}())

function memory_stats(dev::HIPDevice = AMDGPU.device())
    ms = MEMORY_STATS.payload
    did = HIP.device_id(dev)

    stats = get(ms, did, nothing)
    stats ≡ nothing || return stats

    Base.@lock MEMORY_STATS.lock begin
        get!(() -> MemoryStats(), ms, did)
    end
end

function account!(stats::MemoryStats, bytes::Integer)
    Base.@atomic stats.live += bytes
end

# Stats for memory that isn't backed by a device pool (unified, host).
# These allocations can be migrated by the driver and are sized against
# system RAM rather than GPU memory, so we track them globally.
const HOST_STATS = MemoryStats()
host_stats() = HOST_STATS

const EAGER_GC::Ref{Bool} = Ref{Bool}(@load_preference("eager_gc", true))

function eager_gc!(flag::Bool)
    global EAGER_GC[] = flag
    @set_preferences!("eager_gc" => flag)
end

function maybe_collect(; blocking::Bool = false)
    EAGER_GC[] || return

    stats = memory_stats()
    current_time = time()

    if current_time - stats.last_updated > 10
        # Use hard memory limit if set.
        max_size = hard_memory_limit()
        # Otherwise estimate from current usage.
        if max_size == typemax(UInt64)
            dev = device()
            pool = Mem.pool_create(dev)
            free_mem = try
                free()
            catch e
                if e isa HIPError
                    @warn "Failed to query amount of `free()` memory. Disabling eager GC."
                    EAGER_GC[] = false
                    return
                else
                    rethrow(e)
                end
            end
            max_size = free_mem + stats.live +
                (HIP.reserved_memory(pool) - HIP.used_memory(pool))
        end

        Base.@atomic stats.size = max_size
        Base.@atomic stats.last_updated = current_time
    end

    # Similarly re-estimate the host memory budget for unified/host allocations.
    # `Sys.total_memory()` is cgroup-aware, so this does the right thing in containers.
    if current_time - HOST_STATS.last_updated > 10
        Base.@atomic HOST_STATS.size = Int(Sys.total_memory())
        Base.@atomic HOST_STATS.last_updated = current_time
    end

    # Compute pressure for both pools and operate on whichever is dominant.
    device_pressure = stats.size > 0 ? stats.live / stats.size : 0.0
    host_pressure = HOST_STATS.size > 0 ? HOST_STATS.live / HOST_STATS.size : 0.0
    pressure, dominant = device_pressure ≥ host_pressure ?
        (device_pressure, stats) : (host_pressure, HOST_STATS)

    min_pressure = blocking ? 0.5 : 0.75
    pressure < min_pressure && return

    # TODO take allocations into account
    #   if pressure is high but we didn't allocate - don't collect
    #   otherwise try hard

    # Check that we don't collect too often.
    gc_rate = dominant.last_gc_time / (current_time - dominant.last_time)
    # Tolerate 5% GC time.
    max_gc_rate = 0.05
    # If freed a lot of memory last time, double max GC rate.
    (dominant.last_freed > 0.1 * dominant.size) && (max_gc_rate *= 2;)
    # Be more aggressive if we are going to block.
    blocking && (max_gc_rate *= 2;)

    # And even more if the pressure is high.
    pressure > 0.9 && (max_gc_rate *= 2;)
    pressure > 0.95 && (max_gc_rate *= 2;)
    gc_rate > max_gc_rate && return

    Base.@atomic stats.last_time = current_time
    Base.@atomic HOST_STATS.last_time = current_time

    # Call the GC. Snapshot live bytes for both pools before/after, since
    # finalizers running during GC may free memory in either.
    pre_gc_live = stats.live
    pre_gc_host_live = HOST_STATS.live
    gc_time = Base.@elapsed GC.gc(false)

    # Update stats.
    Base.@atomic stats.last_freed = pre_gc_live - stats.live
    Base.@atomic HOST_STATS.last_freed = pre_gc_host_live - HOST_STATS.live
    Base.@atomic stats.last_gc_time = 0.75 * stats.last_gc_time + 0.25 * gc_time
    Base.@atomic HOST_STATS.last_gc_time = 0.75 * HOST_STATS.last_gc_time + 0.25 * gc_time
    return
end


## pool activity tracking

const POOL_STATUS = AMDGPU.LockedObject(Dict{Int, Ref{Bool}}())

function pool_mark(dev::HIPDevice)
    ps = POOL_STATUS.payload
    did = HIP.device_id(dev)
    status = get(ps, did, nothing)
    status === nothing && return nothing
    return status[]
end

function pool_mark!(dev::HIPDevice, val::Bool)
    ps = POOL_STATUS.payload
    did = HIP.device_id(dev)
    box = get(ps, did, nothing)
    if box === nothing
        Base.@lock POOL_STATUS.lock begin
            box = get!(ps, did) do
                Ref{Bool}(val)
            end
        end
    end
    box[] = val
    return
end


## reclaim hooks

"""
    reclaim_hooks

A list of callables that are invoked when memory needs to be reclaimed.
Downstream packages can push functions into this list to free cached resources
(e.g., workspace buffers, FFT plans, etc.) when GPU memory is scarce.
"""
const reclaim_hooks = Any[]


## pool cleanup

const _pool_cleanup_task = Ref{Task}()

function pool_cleanup()
    idle_counters = Dict{Int, Int}()
    while true
        try
            sleep(60)
        catch ex
            if ex isa EOFError
                break
            else
                rethrow()
            end
        end

        for dev in HIP.devices()
            did = HIP.device_id(dev)
            status = pool_mark(dev)
            status === nothing && continue

            if status
                idle_counters[did] = 0
            else
                idle_counters[did] = get(idle_counters, did, 0) + 1
            end
            pool_mark!(dev, false)

            if get(idle_counters, did, 0) >= 5
                HIP.device!(dev) do
                    reclaim()
                end
            end
        end
    end
end


## reclaim

"""
    reclaim([sz=typemax(Int)])

Reclaims `sz` bytes of cached memory. Use this to free GPU memory before
calling into functionality that does not use the memory pool. Returns the
number of bytes actually reclaimed.
"""
function reclaim(sz::Int=typemax(Int))
    dev = AMDGPU.device()
    for hook in reclaim_hooks
        hook()
    end
    HIP.device_synchronize()
    pool = Mem.pool_create(dev)
    before = HIP.reserved_memory(pool)
    HIP.trim(pool)
    after = HIP.reserved_memory(pool)
    return Int(before - after)
end


## pool status & queries

"""
    used_memory()

Returns the amount of memory from the HIP memory pool that is currently
in use by the application.
"""
function used_memory()
    pool = Mem.pool_create(AMDGPU.device())
    Int(HIP.used_memory(pool))
end

"""
    cached_memory()

Returns the amount of backing memory currently allocated (reserved) for the
HIP memory pool.
"""
function cached_memory()
    pool = Mem.pool_create(AMDGPU.device())
    Int(HIP.reserved_memory(pool))
end

"""
    pool_status([io=stdout])

Report to `io` on the memory status of the current GPU and the active memory pool.
"""
function pool_status(io::IO=stdout)
    free_bytes, total_bytes = info()
    used_bytes = total_bytes - free_bytes
    used_ratio = used_bytes / total_bytes
    @printf(io, "Effective GPU memory usage: %.2f%% (%s/%s)\n",
            100*used_ratio, Base.format_bytes(used_bytes),
            Base.format_bytes(total_bytes))

    pool = Mem.pool_create(AMDGPU.device())
    pool_used = HIP.used_memory(pool)
    pool_reserved = HIP.reserved_memory(pool)
    @printf(io, "Memory pool usage: %s (%s reserved)\n",
            Base.format_bytes(pool_used),
            Base.format_bytes(pool_reserved))

    hard_limit = hard_memory_limit()
    soft_limit = soft_memory_limit()
    if hard_limit != typemax(UInt64) || soft_limit != typemax(UInt64)
        print(io, "Memory limit: ")
        parts = String[]
        if soft_limit != typemax(UInt64)
            push!(parts, "soft = $(Base.format_bytes(soft_limit))")
        end
        if hard_limit != typemax(UInt64)
            push!(parts, "hard = $(Base.format_bytes(hard_limit))")
        end
        println(io, join(parts, ", "))
    end
end


# to safely use allocated memory across tasks and devices, we don't simply return raw
# memory objects, but wrap them in a manager that ensures synchronization and ownership.
mutable struct Managed{M}
    const mem::M
    const lock::ReentrantLock

    # which stream is currently using the memory.
    stream::HIPStream

    # whether accessing this memory can cause implicit synchronization
    synchronizing::Bool

    # whether there are outstanding operations that haven't been synchronized
    dirty::Bool

    # whether the memory has been captured in a way that makes the dirty bit unreliable
    captured::Bool

    function Managed(mem; stream=AMDGPU.stream(), synchronizing=true,
                     dirty=true, captured=false)
        # NOTE: memory starts as dirty, because stream-ordered allocations are only
        #       guaranteed to be physically allocated at a synchronization event.
        new{typeof(mem)}(mem, ReentrantLock(), stream, synchronizing, dirty, captured)
    end
end

Base.sizeof(m::Managed) = sizeof(m.mem)

# wait for the current owner of memory to finish processing
function synchronize(managed::Managed)
    Base.@lock managed.lock begin
        synchronize(managed.stream)
        managed.dirty = false
        return
    end
end

function maybe_synchronize(managed::Managed)
    Base.@lock managed.lock begin
        if managed.synchronizing && (managed.dirty || managed.captured)
            synchronize(managed)
        end
        return
    end
end

# Transfer stream ownership of an allocation and mark it dirty in anticipation of a
# device-side operation. The caller must hold `managed.lock` until that operation has
# been submitted to `stream`, so the recorded owner cannot become visible before its
# submission.
function take_ownership!(managed::Managed{M}; stream::HIPStream=AMDGPU.stream()) where M
    sizeof(managed) == 0 && return managed

    # accessing memory during stream capture: taint the memory so we always synchronize
    if HIP.is_capturing(stream)
        managed.captured = true
    end

    # TODO handle access on another device
    # if M <: Mem.HIPBuffer && managed.mem.ctx != tls.ctx
    #     # Enable peer-to-peer access.
    # end

    # accessing memory on another stream: ensure the data is ready and take ownership
    if managed.stream != stream
        maybe_synchronize(managed)
        managed.stream = stream
    end

    # prefetch unified memory as we're likely to use it on the GPU
    if M <: Mem.UnifiedBuffer
        can_prefetch = !HIP.is_capturing(stream)
        can_prefetch &= HIP.attribute(stream.device,
            HIP.hipDeviceAttributeConcurrentManagedAccess) == 1
        can_prefetch &= HIP.ndevices() == 1
        can_prefetch && Mem.prefetch(managed.mem; device=stream.device, stream)
    end

    managed.dirty = true
    return managed
end

function lock_managed(managed::AbstractVector{<:Managed})
    locked = unique(managed)
    sort!(locked; by=m -> objectid(m.lock))
    foreach(m -> lock(m.lock), locked)
    return locked
end

function unlock_managed(locked::AbstractVector{<:Managed})
    foreach(m -> unlock(m.lock), Iterators.reverse(locked))
    return
end

function with_managed(f::F, managed::AbstractVector{<:Managed};
                      stream::HIPStream=AMDGPU.stream()) where {F}
    locked = lock_managed(managed)
    try
        foreach(m -> take_ownership!(m; stream), locked)
        return f()
    finally
        unlock_managed(locked)
    end
end

# NOTE: unlike CUDA.jl, AMDGPU.jl does not have a dedicated device pointer type, so
#       `convert(Ptr, managed)` returns a device-accessible pointer. Use
#       [`host_pointer`](@ref) to get a CPU-accessible pointer instead.
function Base.convert(::Type{Ptr{T}}, managed::Managed{M}) where {T, M}
    Base.@lock managed.lock begin
        # let null pointers pass through as-is
        ptr = convert(Ptr{T}, Mem.device_ptr(managed.mem))
        ptr == C_NULL && return ptr

        take_ownership!(managed)
        return ptr
    end
end

"""
    host_pointer(::Type{Ptr{T}}, managed::Managed)

Return a CPU-accessible pointer to the memory managed by `managed`, making sure
any outstanding device-side operations have finished. Only supported for host
and unified memory.
"""
function host_pointer(::Type{Ptr{T}}, managed::Managed{M}) where {T, M}
    Base.@lock managed.lock begin
        # let null pointers pass through as-is
        ptr = convert(Ptr{T}, managed.mem)
        ptr == C_NULL && return ptr

        # accessing memory on the CPU: only allowed for host or unified allocations
        if M <: Mem.HIPBuffer
            throw(ArgumentError(
                """cannot take the CPU address of GPU memory.

                   You are probably falling back to or otherwise calling CPU functionality
                   with GPU array inputs. This is not supported by regular device memory;
                   ensure this operation is supported by AMDGPU.jl, and if it isn't, try to
                   avoid it or rephrase it in terms of supported operations. Alternatively,
                   you can consider using GPU arrays backed by unified memory by
                   allocating using `roc(...; unified=true)`."""))
        end

        # make sure any work on the memory has finished.
        maybe_synchronize(managed)
        return convert(Ptr{T}, Mem.host_ptr(managed.mem))
    end
end

# TODO workaround until we have HIPPtr
function Base.convert(::Type{Mem.AbstractAMDBuffer}, managed::Managed{M}) where M
    Base.@lock managed.lock begin
        take_ownership!(managed)
        return managed.mem
    end
end

function pool_alloc(::Type{B}, bytesize) where B
    maybe_collect()
    time = Base.@elapsed begin
        s = AMDGPU.stream()
        managed = Managed(B(bytesize; stream=s); stream=s, captured=AMDGPU.is_capturing())
    end

    Base.@atomic alloc_stats.alloc_count += 1
    Base.@atomic alloc_stats.alloc_bytes += bytesize
    Base.@atomic alloc_stats.total_time += time

    pool_mark!(AMDGPU.device(), true)

    if isinteractive() && !isassigned(_pool_cleanup_task)
        _pool_cleanup_task[] = errormonitor(Threads.@spawn pool_cleanup())
    end

    return managed
end

function pool_free(managed::Managed{M}) where M
    sz = Int(sizeof(managed.mem))
    sz == 0 && return

    try
        time = Base.@elapsed Base.@lock managed.lock begin
            _pool_free(managed.mem, managed.stream)
        end
        Base.@atomic alloc_stats.free_count += 1
        Base.@atomic alloc_stats.free_bytes += sz
        Base.@atomic alloc_stats.total_time += time
    catch ex
        Base.showerror_nostdio(ex,
            "WARNING: Error while freeing $(Base.format_bytes(sz)) of GPU memory")
        Base.show_backtrace(Core.stdout, catch_backtrace())
        Core.println()
    end
    return
end

function _pool_free(buf, stream::HIPStream)
    if !HIP.isvalid(stream)
        stream = AMDGPU.default_stream()
    end
    HIP.context!(() -> Mem.free(buf; stream), buf.ctx)
end
