include("reversediffx.jl")

import Distributions: Gamma
using .ReverseDiffX
using .ReverseDiffX: RTR, RTV, RTM

Gamma(α::RTR, θ::Real; check_args=true) = pgamma(α, θ, check_args = check_args)
Gamma(α::Real, θ::RTR; check_args=true) = pgamma(α, θ, check_args = check_args)
Gamma(α::RTR, θ::RTR; check_args=true) = pgamma(α, θ, check_args = check_args)
pgamma(α, θ; check_args=true) = Gamma(promote(α, θ)..., check_args = check_args)
Gamma(α::T; check_args=true) where {T <: RTR} = Gamma(α, one(T), check_args = check_args)
function Gamma(α::T, θ::T; check_args=true) where {T <: RTR}
    check_args && Distributions.@check_args(Gamma, α > zero(α) && θ > zero(θ))
    return Gamma{T}(α, θ)
end

# Work around to stop TrackedReal of Inf and -Inf from producing NaN in the derivative
function Base.minimum(d::LocationScale{T}) where {T <: RTR}
    if isfinite(minimum(d.ρ))
        return d.μ + d.σ * minimum(d.ρ)
    else
        return convert(T, ReverseDiff.@skip(minimum)(d.ρ))
    end
end
function Base.maximum(d::LocationScale{T}) where {T <: RTR}
    if isfinite(minimum(d.ρ))
        return d.μ + d.σ * maximum(d.ρ)
    else
        return convert(T, ReverseDiff.@skip(maximum)(d.ρ))
    end
end

for T in (:RTV, :RTM)
    @eval begin
        function Distributions.logpdf(d::MvNormal{<:Any, <:PDMats.ScalMat}, x::$T)
            logpdf(TuringScalMvNormal(d.μ, d.Σ.value), x)
        end
        function Distributions.logpdf(d::MvNormal{<:Any, <:PDMats.PDiagMat}, x::$T)
            logpdf(TuringDiagMvNormal(d.μ, d.Σ.diag), x)
        end
        function Distributions.logpdf(d::MvNormal{<:Any, <:PDMats.PDMat}, x::$T)
            logpdf(TuringDenseMvNormal(d.μ, d.Σ.chol), x)
        end
        
        function Distributions.logpdf(d::MvLogNormal{<:Any, <:PDMats.ScalMat}, x::$T)
            logpdf(TuringMvLogNormal(TuringScalMvNormal(d.normal.μ, d.normal.Σ.value)), x)
        end
        function Distributions.logpdf(d::MvLogNormal{<:Any, <:PDMats.PDiagMat}, x::$T)
            logpdf(TuringMvLogNormal(TuringDiagMvNormal(d.normal.μ, d.normal.Σ.diag)), x)
        end
        function Distributions.logpdf(d::MvLogNormal{<:Any, <:PDMats.PDMat}, x::$T)
            logpdf(TuringMvLogNormal(TuringDenseMvNormal(d.normal.μ, d.normal.Σ.chol)), x)
        end
    end
end

# zero mean, dense covariance
MvNormal(A::RTM) = TuringMvNormal(A)

# zero mean, diagonal covariance
MvNormal(σ::RTV) = TuringMvNormal(σ)

# dense mean, dense covariance
MvNormal(m::AbstractVector{<:Real}, A::RTM{<:Real}) = TuringMvNormal(m, A)
MvNormal(m::RTV{<:Real}, A::Matrix{<:Real}) = TuringMvNormal(m, A)
MvNormal(m::RTV{<:Real}, A::RTM{<:Real}) = TuringMvNormal(m, A)

# dense mean, diagonal covariance
function MvNormal(
    m::RTV{<:Real},
    D::Diagonal{<:RTR, <:RTV{<:Real}},
)
    return TuringMvNormal(m, D)
end
function MvNormal(
    m::AbstractVector{<:Real},
    D::Diagonal{<:RTR, <:RTV{<:Real}},
)
    return TuringMvNormal(m, D)
end
function MvNormal(
    m::RTV{<:Real},
    D::Diagonal{T, <:AbstractVector{T}} where {T<:Real},
)
    return TuringMvNormal(m, D)
end

# dense mean, diagonal covariance
MvNormal(m::RTV{<:Real}, σ::RTV{<:Real}) = TuringMvNormal(m, σ)
MvNormal(m::RTV{<:Real}, σ::AbstractVector{<:Real}) = TuringMvNormal(m, σ)
MvNormal(m::RTV{<:Real}, σ::Vector{<:Real}) = TuringMvNormal(m, σ)
MvNormal(m::AbstractVector{<:Real}, σ::RTV{<:Real}) = TuringMvNormal(m, σ)

# dense mean, constant variance
MvNormal(m::RTV{<:Real}, σ::RTR) = TuringMvNormal(m, σ)
MvNormal(m::RTV{<:Real}, σ::Real) = TuringMvNormal(m, σ)
MvNormal(m::AbstractVector{<:Real}, σ::RTR) = TuringMvNormal(m, σ)

# dense mean, constant variance
function MvNormal(m::RTV{<:Real}, A::UniformScaling{<:RTR})
    return TuringMvNormal(m, A)
end
function MvNormal(m::AbstractVector{<:Real}, A::UniformScaling{<:RTR})
    return TuringMvNormal(m, A)
end
function MvNormal(m::RTV{<:Real}, A::UniformScaling{<:Real})
    return TuringMvNormal(m, A)
end

# zero mean,, constant variance
MvNormal(d::Int, σ::RTR) = TuringMvNormal(d, σ)

# zero mean, dense covariance
MvLogNormal(A::RTM) = TuringMvLogNormal(TuringMvNormal(A))

# zero mean, diagonal covariance
MvLogNormal(σ::RTV) = TuringMvLogNormal(TuringMvNormal(σ))

# dense mean, dense covariance
MvLogNormal(m::RTV{<:Real}, A::RTM{<:Real}) = TuringMvLogNormal(TuringMvNormal(m, A))
MvLogNormal(m::RTV{<:Real}, A::Matrix{<:Real}) = TuringMvLogNormal(TuringMvNormal(m, A))
MvLogNormal(m::AbstractVector{<:Real}, A::RTM{<:Real}) = TuringMvLogNormal(TuringMvNormal(m, A))

# dense mean, diagonal covariance
function MvLogNormal(
    m::RTV{<:Real},
    D::Diagonal{<:RTR, <:RTV{<:Real}},
)
    return TuringMvLogNormal(TuringMvNormal(m, D))
end
function MvLogNormal(
    m::AbstractVector{<:Real},
    D::Diagonal{<:RTR, <:RTV{<:Real}},
)
    return TuringMvLogNormal(TuringMvNormal(m, D))
end
function MvLogNormal(
    m::RTV{<:Real},
    D::Diagonal{T, <:AbstractVector{T}} where {T<:Real},
)
    return TuringMvLogNormal(TuringMvNormal(m, D))
end

# dense mean, diagonal covariance
MvLogNormal(m::RTV{<:Real}, σ::RTV{<:Real}) = TuringMvLogNormal(TuringMvNormal(m, σ))
MvLogNormal(m::RTV{<:Real}, σ::AbstractVector{<:Real}) = TuringMvLogNormal(TuringMvNormal(m, σ))
MvLogNormal(m::RTV{<:Real}, σ::Vector{<:Real}) = TuringMvLogNormal(TuringMvNormal(m, σ))
MvLogNormal(m::AbstractVector{<:Real}, σ::RTV{<:Real}) = TuringMvLogNormal(TuringMvNormal(m, σ))

# dense mean, constant variance
function MvLogNormal(m::RTV{<:Real}, σ::RTR)
    return TuringMvLogNormal(TuringMvNormal(m, σ))
end
function MvLogNormal(m::RTV{<:Real}, σ::Real)
    return TuringMvLogNormal(TuringMvNormal(m, σ))
end
function MvLogNormal(m::AbstractVector{<:Real}, σ::RTR)
    return TuringMvLogNormal(TuringMvNormal(m, σ))
end

# dense mean, constant variance
function MvLogNormal(m::RTV{<:Real}, A::UniformScaling{<:RTR})
    return TuringMvLogNormal(TuringMvNormal(m, A))
end
function MvLogNormal(m::AbstractVector{<:Real}, A::UniformScaling{<:RTR})
    return TuringMvLogNormal(TuringMvNormal(m, A))
end
function MvLogNormal(m::RTV{<:Real}, A::UniformScaling{<:Real})
    return TuringMvLogNormal(TuringMvNormal(m, A))
end

# zero mean,, constant variance
MvLogNormal(d::Int, σ::RTR) = TuringMvLogNormal(TuringMvNormal(d, σ))

Distributions.Dirichlet(alpha::RTV) = TuringDirichlet(alpha)
Distributions.Dirichlet(d::Integer, alpha::RTR) = TuringDirichlet(d, alpha)

function Distributions.logpdf(d::MatrixBeta, X::AbstractArray{<:RTM{<:Real}})
    return mapvcat(x -> logpdf(d, x), X)
end

Distributions.Wishart(df::RTR, S::Matrix{<:Real}) = TuringWishart(df, S)
Distributions.Wishart(df::RTR, S::AbstractMatrix{<:Real}) = TuringWishart(df, S)
Distributions.Wishart(df::Real, S::RTM) = TuringWishart(df, S)
Distributions.Wishart(df::RTR, S::RTM) = TuringWishart(df, S)

Distributions.InverseWishart(df::RTR, S::Matrix{<:Real}) = TuringInverseWishart(df, S)
Distributions.InverseWishart(df::RTR, S::AbstractMatrix{<:Real}) = TuringInverseWishart(df, S)
Distributions.InverseWishart(df::Real, S::RTM) = TuringInverseWishart(df, S)
Distributions.InverseWishart(df::RTR, S::RTM) = TuringInverseWishart(df, S)

function Distributions.logpdf(d::Wishart, X::RTM)
    return logpdf(TuringWishart(d), X)
end
function Distributions.logpdf(d::Wishart, X::AbstractArray{<:RTM})
    return logpdf(TuringWishart(d), X)
end

function Distributions.logpdf(d::InverseWishart, X::RTM)
    return logpdf(TuringInverseWishart(d), X)
end
function Distributions.logpdf(d::InverseWishart, X::AbstractArray{<:RTM})
    return logpdf(TuringInverseWishart(d), X)
end
