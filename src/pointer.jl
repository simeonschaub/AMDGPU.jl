# AMDGPU pointer types

export ROCPtr, ROC_NULL, PtrOrROCPtr, ROCArrayPtr, ROCRef


#
# AMDGPU device pointer
#

"""
    ROCPtr{T}

A memory address that refers to data of type `T` that is accessible from the GPU. A `ROCPtr`
is ABI compatible with regular `Ptr` objects, e.g. it can be used to `ccall` a function that
expects a `Ptr` to GPU memory, but it prevents erroneous conversions between the two.
"""
ROCPtr

if sizeof(Ptr{Cvoid}) == 8
    primitive type ROCPtr{T} 64 end
else
    primitive type ROCPtr{T} 32 end
end

# constructor
ROCPtr{T}(x::Union{Int,UInt,ROCPtr}) where {T} = Base.bitcast(ROCPtr{T}, x)

const ROC_NULL = ROCPtr{Cvoid}(0)


## getters

Base.eltype(::Type{<:ROCPtr{T}}) where {T} = T


## conversions

# to and from integers
## pointer to integer
Base.convert(::Type{T}, x::ROCPtr) where {T<:Integer} = T(UInt(x))
## integer to pointer
Base.convert(::Type{ROCPtr{T}}, x::Union{Int,UInt}) where {T} = ROCPtr{T}(x)
Base.Int(x::ROCPtr)  = Base.bitcast(Int, x)
Base.UInt(x::ROCPtr) = Base.bitcast(UInt, x)

# between regular and AMDGPU pointers
Base.convert(::Type{<:Ptr}, p::ROCPtr) =
    throw(ArgumentError("cannot convert a GPU pointer to a CPU pointer"))

# between AMDGPU pointers
Base.convert(::Type{ROCPtr{T}}, p::ROCPtr) where {T} = Base.bitcast(ROCPtr{T}, p)

# defer conversions to unsafe_convert
Base.cconvert(::Type{<:ROCPtr}, x) = x

# fallback for unsafe_convert
Base.unsafe_convert(::Type{P}, x::ROCPtr) where {P<:ROCPtr} = convert(P, x)

# in HIP, contrary to CUDA, ROCPtr can be converted to Ptr
Base.unsafe_convert(::Type{Ptr{T}}, x::ROCPtr) where {T} = Base.bitcast(Ptr{T}, x)

# from arrays
Base.unsafe_convert(::Type{ROCPtr{S}}, a::AbstractArray{T}) where {S,T} =
    convert(ROCPtr{S}, Base.unsafe_convert(ROCPtr{T}, a))
Base.unsafe_convert(::Type{ROCPtr{T}}, a::AbstractArray{T}) where {T} =
    error("conversion to pointer not defined for $(typeof(a))")

## limited pointer arithmetic & comparison

Base.isequal(x::ROCPtr, y::ROCPtr) = (x === y)
Base.isless(x::ROCPtr{T}, y::ROCPtr{T}) where {T} = x < y

Base.:(==)(x::ROCPtr, y::ROCPtr) = UInt(x) == UInt(y)
Base.:(<)(x::ROCPtr,  y::ROCPtr) = UInt(x) < UInt(y)
Base.:(-)(x::ROCPtr,  y::ROCPtr) = UInt(x) - UInt(y)

if VERSION >= v"1.12.0-DEV.225"
Base.:(+)(x::ROCPtr{T}, y::Integer) where T =
    reinterpret(ROCPtr{T}, Base.add_ptr(reinterpret(Ptr{T}, x), (y % UInt) % UInt))
Base.:(-)(x::ROCPtr{T}, y::Integer) where T =
    reinterpret(ROCPtr{T}, Base.sub_ptr(reinterpret(Ptr{T}, x), (y % UInt) % UInt))
else
Base.:(+)(x::ROCPtr, y::Integer) = oftype(x, Base.add_ptr(UInt(x), (y % UInt) % UInt))
Base.:(-)(x::ROCPtr, y::Integer) = oftype(x, Base.sub_ptr(UInt(x), (y % UInt) % UInt))
end
Base.:(+)(x::Integer, y::ROCPtr) = y + x



#
# Host or device pointer
#

"""
    PtrOrROCPtr{T}

A special pointer type, ABI-compatible with both `Ptr` and `ROCPtr`, for use in `ccall`
expressions to convert values to either a GPU or a CPU type (in that order). This is
required for HIP APIs which accept pointers that either point to host or device memory.
"""
PtrOrROCPtr

if sizeof(Ptr{Cvoid}) == 8
    primitive type PtrOrROCPtr{T} 64 end
else
    primitive type PtrOrROCPtr{T} 32 end
end

function Base.cconvert(::Type{PtrOrROCPtr{T}}, val) where {T}
    # `cconvert` is always implemented for both `Ptr` and `ROCPtr`, so pick the first result
    # that has done an actual conversion

    gpu_val = Base.cconvert(ROCPtr{T}, val)
    if gpu_val !== val
        return gpu_val
    end

    cpu_val = Base.cconvert(Ptr{T}, val)
    if cpu_val !== val
        return cpu_val
    end

    return val
end

function Base.unsafe_convert(::Type{PtrOrROCPtr{T}}, val) where {T}
    ptr = if Core.Compiler.return_type(Base.unsafe_convert,
                                       Tuple{Type{Ptr{T}}, typeof(val)}) !== Union{}
        Base.unsafe_convert(Ptr{T}, val)
    elseif Core.Compiler.return_type(Base.unsafe_convert,
                                     Tuple{Type{ROCPtr{T}}, typeof(val)}) !== Union{}
        Base.unsafe_convert(ROCPtr{T}, val)
    else
        throw(ArgumentError("cannot convert to either a CPU or GPU pointer"))
    end

    return Base.bitcast(PtrOrROCPtr{T}, ptr)
end

# avoid ambiguities when passing PtrOrROCPtr instances
Base.unsafe_convert(::Type{PtrOrROCPtr{T}}, x::PtrOrROCPtr{T}) where {T} = x


#
# AMDGPU array pointer
#

if sizeof(Ptr{Cvoid}) == 8
    primitive type ROCArrayPtr{T} 64 end
else
    primitive type ROCArrayPtr{T} 32 end
end

# constructor
ROCArrayPtr{T}(x::Union{Int,UInt,ROCArrayPtr}) where {T} = Base.bitcast(ROCArrayPtr{T}, x)


## getters

Base.eltype(::Type{<:ROCArrayPtr{T}}) where {T} = T


## conversions

# to and from integers
## pointer to integer
Base.convert(::Type{T}, x::ROCArrayPtr) where {T<:Integer} = T(UInt(x))
## integer to pointer
Base.convert(::Type{ROCArrayPtr{T}}, x::Union{Int,UInt}) where {T} = ROCArrayPtr{T}(x)
Base.Int(x::ROCArrayPtr)  = Base.bitcast(Int, x)
Base.UInt(x::ROCArrayPtr) = Base.bitcast(UInt, x)

# between regular and AMDGPU pointers
Base.convert(::Type{<:Ptr}, p::ROCArrayPtr) =
    throw(ArgumentError("cannot convert a GPU array pointer to a CPU pointer"))

# between AMDGPU array pointers
Base.convert(::Type{ROCArrayPtr{T}}, p::ROCArrayPtr) where {T} = Base.bitcast(ROCArrayPtr{T}, p)

# defer conversions to unsafe_convert
Base.cconvert(::Type{<:ROCArrayPtr}, x) = x

# fallback for unsafe_convert
Base.unsafe_convert(::Type{P}, x::ROCArrayPtr) where {P<:ROCArrayPtr} = convert(P, x)


## limited pointer arithmetic & comparison

Base.isequal(x::ROCArrayPtr, y::ROCArrayPtr) = (x === y)
Base.isless(x::ROCArrayPtr{T}, y::ROCArrayPtr{T}) where {T} = x < y

Base.:(==)(x::ROCArrayPtr, y::ROCArrayPtr) = UInt(x) == UInt(y)
Base.:(<)(x::ROCArrayPtr,  y::ROCArrayPtr) = UInt(x) < UInt(y)
Base.:(-)(x::ROCArrayPtr,  y::ROCArrayPtr) = UInt(x) - UInt(y)

if VERSION >= v"1.12.0-DEV.225"
Base.:(+)(x::ROCArrayPtr{T}, y::Integer) where T =
    reinterpret(ROCArrayPtr{T}, Base.add_ptr(reinterpret(Ptr{T}, x), (y % UInt) % UInt))
Base.:(-)(x::ROCArrayPtr{T}, y::Integer) where T =
    reinterpret(ROCArrayPtr{T}, Base.sub_ptr(reinterpret(Ptr{T}, x), (y % UInt) % UInt))
else
Base.:(+)(x::ROCArrayPtr, y::Integer) = oftype(x, Base.add_ptr(UInt(x), (y % UInt) % UInt))
Base.:(-)(x::ROCArrayPtr, y::Integer) = oftype(x, Base.sub_ptr(UInt(x), (y % UInt) % UInt))
end
Base.:(+)(x::Integer, y::ROCArrayPtr) = y + x



#
# AMDGPU reference objects (forward declaration)
#

if sizeof(Ptr{Cvoid}) == 8
    primitive type ROCRef{T} 64 end
else
    primitive type ROCRef{T} 32 end
end
