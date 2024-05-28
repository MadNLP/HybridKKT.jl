"""
    Kernels for data transfer.
"""

#=
    Default implementation.
=#

"Implement `dest .= src[idx]`."
function index_copy!(dest::AbstractVector{T}, src::AbstractVector{T}, idx::AbstractVector{Ti}) where {T, Ti<:Integer}
    @assert length(dest) == length(idx)
    @inbounds for i in eachindex(idx)
        dest[i] = src[idx[i]]
    end
end

"Implement `dest[idx] .= src`."
function index_copy!(dest::AbstractVector{T}, idx::AbstractVector{Ti}, src::AbstractVector{T}) where {T, Ti<:Integer}
    @assert length(src) == length(idx)
    @inbounds for i in eachindex(idx)
        dest[idx[i]] = src[i]
    end
end

"Implement `dest[idx] .= val`."
function fixed!(dest::AbstractVector{T}, idx::AbstractVector{Ti}, val::T) where {T, Ti<:Integer}
    @inbounds for i in idx
        dest[i] = val
    end
end

function transfer_coef!(G::SparseMatrixCSC, map::Vector{Int}, coefs::Vector{Tv}, ind_eq) where Tv
    valsG = nonzeros(G)
    fill!(valsG, zero(Tv))
    for k in 1:length(map)
        valsG[map[k]] += coefs[ind_eq[k]]
    end
    return
end

#=
    KA implementation.
=#

@kernel function _copy_index_from_kernel!(dest, src, idx)
    i = @index(Global, Linear)
    @inbounds dest[i] = src[idx[i]]
end
function index_copy!(dest::CuVector{T}, src::CuVector{T}, idx::CuVector{Ti}) where {T, Ti<:Integer}
    @assert length(dest) == length(idx)
    if length(dest) > 0
        ndrange = (length(dest),)
        _copy_index_from_kernel!(CUDABackend())(dest, src, idx; ndrange=ndrange)
        KernelAbstractions.synchronize(CUDABackend())
    end
end

@kernel function _copy_index_to_kernel!(dest, src, idx)
    i = @index(Global, Linear)
    @inbounds dest[idx[i]] = src[i]
end
function index_copy!(dest::CuVector{T}, idx::CuVector{Ti}, src::CuVector{T}) where {T, Ti<:Integer}
    @assert length(src) == length(idx)
    if length(src) > 0
        ndrange = (length(src),)
        _copy_index_to_kernel!(CUDABackend())(dest, src, idx; ndrange=ndrange)
        KernelAbstractions.synchronize(CUDABackend())
    end
end

@kernel function _fixed_kernel!(dest, idx, val)
    i = @index(Global, Linear)
    dest[idx[i]] = val
end
function fixed!(dest::CuVector{T}, idx::CuVector{Ti}, val::T) where {T, Ti<:Integer}
    length(idx) == 0 && return
    _fixed_kernel!(CUDABackend())(dest, idx, val; ndrange=length(idx))
    KernelAbstractions.synchronize(CUDABackend())
end

@kernel function _transfer_coef_kernel!(valsG, to_map, valJ, fr_map)
    k = @index(Global, Linear)
    @inbounds begin
        Atomix.@atomic valsG[to_map[k]] += valJ[fr_map[k]]
    end
end
function transfer_coef!(G::CUSPARSE.CuSparseMatrixCSC, map::CuVector{Int}, coefs::CuVector{Tv}, ind_eq) where Tv
    valsG = nonzeros(G)
    fill!(valsG, zero(Tv))
    _transfer_coef_kernel!(CUDABackend())(valsG, map, coefs, ind_eq; ndrange=length(map))
    return
end

