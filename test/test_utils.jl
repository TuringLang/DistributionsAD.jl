# Create positive definite matrix
to_posdef(A::AbstractMatrix) = A * A' + I
to_posdef_diagonal(a::AbstractVector) = Diagonal(a .^ 2 .+ 1)

# Create vectors in probability simplex.
function to_simplex(x::AbstractArray)
    max = maximum(x; dims = 1)
    y = exp.(x .- max)
    y ./= sum(y; dims = 1)
    return y
end
to_simplex(x::AbstractArray{<:AbstractArray}) = to_simplex.(x)

# Utility for testing `adapt_randn`
function test_adapt_randn(rng, x::AbstractVector, ::Type{T}, dims::Int...) where {T}
    Random.seed!(rng, 100)
    y = DistributionsAD.adapt_randn(rng, x, dims...)
    @test y isa Array{T}
    @test size(y) == dims

    Random.seed!(rng, 100)
    @test y == randn(rng, T, dims...)
end
