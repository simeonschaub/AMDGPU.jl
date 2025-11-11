using ..CEnum
using ..GPUCompiler

import ..Adapt
import Preferences: @load_preference, @set_preferences!

import ..HSA
import ..HIP
using ..AMDGPU
using Printf: @printf
using GPUToolbox: LazyInitialized, @enum_without_prefix
using LLVM.Interop: assume

struct Adaptor end

const RT_LOCK = Threads.ReentrantLock()
const RT_EXITING = Ref{Bool}(false)

include("error.jl")
include("dims.jl")
include("memory.jl")
include("state.jl")

include("execution.jl")
include("hip-execution.jl")
