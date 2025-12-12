# global state management
#
# The functionality in this file serves to create a coherent environment to perform
# computations in, with support for Julia constructs like tasks (and executing those on
# multiple threads), using a GPU of your choice (with the ability to reset that device, or
# use different devices on different threads).
#
# this is complicated by the fact that HIP uses thread-bound contexts, while we want to be
# able to switch between tasks executing on the same thread. the current approach involves
# prefixing every HIP API call with code that ensures the HIP thread-bound state is
# identical to Julia's task-local one.

export context, context!, device, device!, device_reset!, deviceid, stream, stream!


## task-local state

@enum MathMode begin
    # use prescribed precision and standardized arithmetic for all calculations.
    # this may serialize operations, and reduce performance.
    PEDANTIC_MATH

    # use at least the required precision, and allow reordering operations for performance.
    DEFAULT_MATH

    # additionally allow downcasting operations for better use of hardware resources.
    # whenever possible the `precision` flag passed to `math_mode!` will be used
    # to constrain those downcasts.
    FAST_MATH
end

# math mode and precision are sticky (once set on a task, inherit to newly created tasks)
const default_math_mode = Ref{Union{Nothing,MathMode}}(nothing)
const default_math_precision = Ref{Union{Nothing,Symbol}}(nothing)

# the default device unitialized tasks will use, set when switching devices.
# this behavior differs from the HIP Runtime, where device 0 is always used.
# this setting won't be used when switching tasks on a pre-initialized thread.
const default_device = Ref{Union{Nothing,HIPDevice}}(nothing)

mutable struct TaskLocalState
    device::HIPDevice
    streams::Vector{Union{Nothing,HIPStream}}
    math_mode::MathMode
    math_precision::Symbol

    function TaskLocalState(dev::HIPDevice=something(default_device[], HIPDevice(0)))
        math_mode = something(default_math_mode[],
                              Base.JLOptions().fast_math==1 ? FAST_MATH : DEFAULT_MATH)
        math_precision = something(default_math_precision[], :TensorFloat32)
        new(dev, Union{Nothing,HIPStream}[nothing for _ in 1:ndevices()],
            math_mode, math_precision)
    end
end

function validate_task_local_state(state::TaskLocalState)
    return state
end

# get or create the task local state, and make sure it's valid
function task_local_state!(args...)
    tls = task_local_storage()
    if haskey(tls, :AMDGPU)
        validate_task_local_state(@inbounds(tls[:AMDGPU])::TaskLocalState)
    else
        # verify that AMDGPU.jl is functional. this doesn't belong here, but since we can't
        # error during `__init__`, we do it here instead as this is the first function
        # that's likely executed when using AMDGPU.jl
        @assert functional()

        tls[:AMDGPU] = TaskLocalState(args...)
    end::TaskLocalState
end

# only get the task local state (it may be invalid!), or return nothing if unitialized
function task_local_state()
    tls = task_local_storage()
    if haskey(tls, :AMDGPU)
        @inbounds(tls[:AMDGPU])
    else
        nothing
    end::Union{TaskLocalState,Nothing}
end

@inline function prepare_hip_state()
    state = task_local_state!()

    dev = Ref{Cint}()
    HIP.hipGetDevice(dev)
    if dev[] != deviceid(state.device)
        HIP.hipSetDevice(deviceid(state.device))
    end

    return
end

# convenience function to get all relevant state
# without querying task local storage multiple times
@inline function active_state()
    # inline to remove unused state properties
    state = task_local_state!()
    return (device=state.device, stream=stream(state),
            math_mode=state.math_mode, math_precision=state.math_precision)
end


## device-based API

"""
    device()::HIPDevice

Get the HIP device for the current thread, similar to how [`context()`](@ref) works
compared to [`current_context()`](@ref).
"""
function device()
    task_local_state!().device
end

"""
    device!(dev::Integer)
    device!(dev::HIPDevice)
    device!(dev) do ... end

Sets `dev` as the current active device for the calling host thread. Devices can be
specified by integer id, or as a `HIPDevice` (slightly faster). Both functions can be used
with do-block syntax, in which case the device is only changed temporarily, without changing
the default device used to initialize new threads or tasks.

Calling this function at the start of a session will make sure HIP is initialized (i.e.,
a primary context will be created and activated).
"""
device!

function device!(dev::HIPDevice, flags=nothing)
    # configure the primary device flags
    if flags !== nothing
        devidx = deviceid(dev)+1
        error("Device flags are no longer supported")
    end

    # make this device the new default
    default_device[] = dev

    # switch device
    state = task_local_state()
    if state === nothing
        task_local_state!(dev)
    else
        state.device = dev
    end
    HIP.hipSetDevice(deviceid(dev))

    dev
end

function device!(f::Function, dev::HIPDevice)
    old_dev = device()
    device!(dev)
    try
        f()
    finally
        device!(old_dev)
    end
end

"""
    device_reset!(dev::HIPDevice=device())

Reset the HIP state associated with a device. This call with release the underlying
context, at which point any objects allocated in that context will be invalidated.

Note that this does not guarantee to free up all memory allocations, as many are not bound
to a context, so it is generally not useful to call this function to free up memory.
"""
function device_reset!(dev::HIPDevice=device())
    # unconditionally reset the primary context (don't just release it),
    # as there might be users outside of AMDGPU.jl
    pctx = HIPPrimaryContext(dev)
    unsafe_reset!(pctx)

    return
end


## integer device-based API

device!(dev::Integer, flags=nothing) = device!(HIPDevice(dev), flags)
device!(f::Function, dev::Integer) = device!(f, HIPDevice(dev))


## math mode

function math_mode!(mode::MathMode; precision=nothing)
    state = task_local_state!()

    state.math_mode = mode
    default_math_mode[] = mode

    if precision !== nothing
        state.math_precision = precision
        default_math_precision[] = precision
    end

    return
end

math_mode() = task_local_state!().math_mode
math_precision() = task_local_state!().math_precision


## streams

"""
    stream()

Get the HIP stream that should be used as the default one for the currently executing task.
"""
@inline function stream(state=task_local_state!())
    # @inline so that it can be DCE'd when unused from active_state
    devidx = deviceid(state.device)+1
    @inbounds if state.streams[devidx] === nothing
        state.streams[devidx] = create_stream()
    else
        state.streams[devidx]::HIPStream
    end
end
@noinline function create_stream()
    stream = HIPStream()

    # register the name of this task
    # XXX: do this when the user has imported ROCTRACER.jl (using weak dependencies?)
    #t = current_task()
    #tptr = pointer_from_objref(current_task())
    #tptrstr = string(convert(UInt, tptr), base=16, pad=Sys.WORD_SIZE>>2)
    #ROCTRACER.roctracerNameHipStreamA(stream, "Task(0x$tptrstr)")

    stream
end

function stream!(stream::HIPStream)
    state = task_local_state!()
    devidx = deviceid(state.device)+1
    state.streams[devidx] = stream
    return
end

function stream!(f::Function, stream::HIPStream)
    state = task_local_state!()
    devidx = deviceid(state.device)+1
    old_stream = state.streams[devidx]
    state.streams[devidx] = stream
    try
        f()
    finally
        state.streams[devidx] = old_stream
    end
end

"""
    stream!(::HIPStream)
    stream!(::HIPStream) do ... end

Change the default HIP stream for the currently executing task, temporarily if using the
do-block version of this function.
"""
stream!


## helpers

"""
    PerDevice{T}()

A helper struct for maintaining per-device state that's lazily initialized and automatically
invalidated when the device is reset. Use `get!(per_device, dev) do ... end` to initialize
and fetch a value.

Mutating or deleting state is not supported. If this is required, use a boxed value, like
a `Ref` or a `Threads.Atomic`.

Furthermore, even though the initialization of this helper, fetching its value for a
given device, and clearing it when the device is reset are all performed in a thread-safe
manner, you should still take care about thread-safety when using the contained value.
For example, if you need to update the value, use atomics; if it's a complex structure like
an array or a dictionary, use additional locks.
"""
struct PerDevice{T}
    lock::ReentrantLock
    values::LazyInitialized{Vector{Union{Nothing,Tuple{HIPDevice,T}}},Nothing}
end

function PerDevice{T}() where {T}
    values = LazyInitialized{Vector{Union{Nothing,Tuple{HIPDevice,T}}}}()
    PerDevice{T}(ReentrantLock(), values)
end

get_values(x::PerDevice{T}) where {T} = get!(x.values) do
    Union{Nothing,Tuple{HIPDevice,T}}[nothing for _ in 1:ndevices()]
end

function Base.get(x::PerDevice, dev::HIPDevice, val)
    y = get_values(x)
    id = deviceid(dev)+1
    @inbounds begin
        # test-lock-test
        if y[id] === nothing || y[id][1] !== dev
            val
        else
            y[id][2]
        end
    end
end

function Base.get!(constructor::F, x::PerDevice{T}, dev::HIPDevice) where {F <: Base.Callable, T}
    y = get_values(x)
    global _y = y
    id = deviceid(dev)+1
    @inbounds begin
        # test-lock-test
        if y[id] === nothing || (y[id]::Tuple)[1] !== dev
            Base.@lock x.lock begin
                if y[id] === nothing || (y[id]::Tuple)[1] !== dev
                    y[id] = (dev, constructor())
                end
            end
        end
        (y[id]::Tuple)[2]
    end
end

Base.length(x::PerDevice) = length(get_values(x))
Base.size(x::PerDevice) = size(get_values(x))
Base.keys(x::PerDevice) = keys(get_values(x))

function Base.show(io::IO, mime::MIME"text/plain", x::PerDevice{T}) where {T}
    print(io, "PerDevice{$T}:")
    for dev in devices()
        print(io, "\n $(deviceid(dev)): ", get(x, dev, "#undef"))
    end
end
