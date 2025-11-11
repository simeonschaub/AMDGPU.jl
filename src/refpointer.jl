# reference objects

abstract type AbstractROCRef{T} <: Ref{T} end

## opaque reference type
##
## we use a concrete ROCRef type that actual references can be (no-op) converted to, without
## actually being a subtype of ROCRef. This is necessary so that `ROCRef` can be used in
## `ccall` signatures; which Base solves by special-casing `Ref` handing in `ccall.cpp`.
# forward declaration in pointer.jl

# general methods for ROCRef{T} type
Base.eltype(x::Type{<:ROCRef{T}}) where {T} = @isdefined(T) ? T : Any

Base.convert(::Type{ROCRef{T}}, x::ROCRef{T}) where {T} = x

# conversion for the actual ccall
Base.unsafe_convert(::Type{ROCRef{T}}, x::ROCRef{T}) where {T} = Base.bitcast(ROCRef{T}, Base.unsafe_convert(ROCPtr{T}, x))
Base.unsafe_convert(::Type{ROCRef{T}}, x) where {T} = Base.bitcast(ROCRef{T}, Base.unsafe_convert(ROCPtr{T}, x))
## avoid double conversions (for compatibility)
Base.unsafe_convert(::Type{ROCPtr{T}}, x::ROCRef{T}) where {T} = x

# ROCRef from literal pointer
Base.convert(::Type{ROCRef{T}}, x::ROCPtr{T}) where {T} = x

# indirect constructors using ROCRef
ROCRef(x::Any) = ROCRefValue(x)
ROCRef{T}(x) where {T} = ROCRefValue{T}(x)
ROCRef{T}() where {T} = ROCRefValue{T}()
Base.convert(::Type{ROCRef{T}}, x) where {T} = ROCRef{T}(x)

# idempotency
Base.convert(::Type{ROCRef{T}}, x::AbstractROCRef{T}) where {T} = x


## reference backed by a single allocation

mutable struct ROCRefValue{T} <: AbstractROCRef{T}
    buf::Managed{Runtime.Mem.DeviceMemory}

    function ROCRefValue{T}() where {T}
        @assert isbitstype(T) "ROCRef only supports bits types"
        buf = pool_alloc(Runtime.Mem.DeviceMemory, Base.aligned_sizeof(T))
        obj = new(buf)
        finalizer(obj) do _
            pool_free(buf)
        end
        return obj
    end
end
function ROCRefValue{T}(x::T) where {T}
    ref = ROCRefValue{T}()
    ref[] = x
    return ref
end
ROCRefValue{T}(x) where {T} = ROCRefValue{T}(convert(T, x))
ROCRefValue(x::T) where {T} = ROCRefValue{T}(x)

Base.unsafe_convert(::Type{ROCPtr{T}}, b::ROCRefValue{T}) where {T} = convert(ROCPtr{T}, b.buf)
Base.unsafe_convert(P::Type{ROCPtr{Any}}, b::ROCRefValue{Any}) = convert(P, b.buf)
Base.unsafe_convert(::Type{ROCPtr{Cvoid}}, b::ROCRefValue{T}) where {T} =
    convert(ROCPtr{Cvoid}, b.buf)

function Base.setindex!(gpu::ROCRefValue{T}, x::T) where {T}
    cpu = Ref(x)
    GC.@preserve cpu begin
        cpu_ptr = Base.unsafe_convert(Ptr{T}, cpu)
        gpu_ptr = Base.unsafe_convert(ROCPtr{T}, gpu)
        unsafe_copyto!(gpu_ptr, cpu_ptr, 1; async=true)
    end
    return gpu
end

function Base.getindex(gpu::ROCRefValue{T}) where {T}
    # synchronize first to maximize time spent executing Julia code
    synchronize(gpu.buf)

    cpu = Ref{T}()
    GC.@preserve cpu begin
        cpu_ptr = Base.unsafe_convert(Ptr{T}, cpu)
        gpu_ptr = Base.unsafe_convert(ROCPtr{T}, gpu)
        unsafe_copyto!(cpu_ptr, gpu_ptr, 1; async=false)
    end
    cpu[]
end

function Base.show(io::IO, x::ROCRefValue{T}) where {T}
    print(io, "ROCRefValue{$T}(")
    print(io, x[])
    print(io, ")")
end


## reference backed by a AMDGPU array at index i

struct ROCRefArray{T,A<:AbstractArray{T}} <: AbstractROCRef{T}
    x::A
    i::Int
    ROCRefArray{T,A}(x,i) where {T,A<:AbstractArray{T}} = new(x,i)
end
ROCRefArray{T}(x::AbstractArray{T}, i::Int=1) where {T} = ROCRefArray{T,typeof(x)}(x, i)
ROCRefArray(x::AbstractArray{T}, i::Int=1) where {T} = ROCRefArray{T}(x, i)

Base.convert(::Type{ROCRef{T}}, x::AbstractArray{T}) where {T} = ROCRefArray(x, 1)
Base.convert(::Type{ROCRef{T}}, x::ROCRefArray{T}) where {T} = x

Base.unsafe_convert(P::Type{ROCPtr{T}}, b::ROCRefArray{T}) where {T} = pointer(b.x, b.i)
Base.unsafe_convert(P::Type{ROCPtr{Any}}, b::ROCRefArray{Any}) = convert(P, pointer(b.x, b.i))
Base.unsafe_convert(::Type{ROCPtr{Cvoid}}, b::ROCRefArray{T}) where {T} =
    convert(ROCPtr{Cvoid}, Base.unsafe_convert(ROCPtr{T}, b))

function Base.setindex!(gpu::ROCRefArray{T}, x::T) where {T}
    cpu = Ref(x)
    GC.@preserve cpu begin
        cpu_ptr = Base.unsafe_convert(Ptr{T}, cpu)
        gpu_ptr = pointer(gpu.x, gpu.i)
        unsafe_copyto!(gpu_ptr, cpu_ptr, 1; async=true)
    end
    return gpu
end

function Base.getindex(gpu::ROCRefArray{T}) where {T}
    # synchronize first to maximize time spent executing Julia code
    synchronize(gpu.x)

    cpu = Ref{T}()
    GC.@preserve cpu begin
        cpu_ptr = Base.unsafe_convert(Ptr{T}, cpu)
        gpu_ptr = pointer(gpu.x, gpu.i)
        unsafe_copyto!(cpu_ptr, gpu_ptr, 1; async=false)
    end
    cpu[]
end

function Base.show(io::IO, x::ROCRefArray{T}) where {T}
    print(io, "ROCRefArray{$T}(")
    print(io, x[])
    print(io, ")")
end
