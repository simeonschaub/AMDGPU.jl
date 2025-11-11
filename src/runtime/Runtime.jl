module Runtime

using ..CEnum
using ..GPUCompiler

import ..Adapt
import Preferences: @load_preference, @set_preferences!

import ..HSA
import ..HIP
using ..AMDGPU
import .HIP: HIPDevice
using Printf: @printf
using GPUToolbox: LazyInitialized
using ..AMDGPU: devices, device_id as deviceid
using LLVM.Interop: assume
ndevices() = length(devices()) + 1

struct Adaptor end

const RT_LOCK = Threads.ReentrantLock()
const RT_EXITING = Ref{Bool}(false)

include("error.jl")
include("dims.jl")
include("pool.jl")
include("state.jl")
include("memory.jl")

include("execution.jl")
include("hip-execution.jl")

end
