module Runtime

using ..CEnum
using ..GPUCompiler

import ..Adapt
import Preferences: @load_preference, @set_preferences!

import ..HSA
import ..HIP
import ..AMDGPU
import ..AMDGPU: LockedObject
import .HIP: HIPDevice

struct Adaptor end

const RT_LOCK = Threads.ReentrantLock()
const RT_EXITING = Ref{Bool}(false)

include("error.jl")
include("dims.jl")
include("memory.jl")

include("execution.jl")
include("hip-execution.jl")

end
