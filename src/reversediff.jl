const RTR = ReverseDiff.TrackedReal
const RTV = ReverseDiff.TrackedVector
const RTM = ReverseDiff.TrackedMatrix
import SpecialFunctions: logbeta
import Distributions: Gamma

# Use ForwardDiff for binomlogpdf and logbeta

ReverseDiff.@forward binomlogpdf(n::Int, p::RTR, x::Int) = begin
    return binomlogpdf(n, p, x)
end

ReverseDiff.@forward logbeta(x::RTR, y::RTR) = begin
    return logbeta(x, y)
end
ReverseDiff.@forward logbeta(x::RTR, y::Real) = begin
    return logbeta(x, y)
end
ReverseDiff.@forward logbeta(x::Real, y::RTR) = begin
    return logbeta(x, y)
end

Gamma(α::RTR, θ::Real; check_args=true) = pgamma(α, θ, check_args = check_args)
Gamma(α::Real, θ::RTR; check_args=true) = pgamma(α, θ, check_args = check_args)
Gamma(α::RTR, θ::RTR; check_args=true) = pgamma(α, θ, check_args = check_args)
pgamma(α, θ; check_args=true) = Gamma(promote(α, θ)..., check_args = check_args)
Gamma(α::T; check_args=true) where {T <: RTR} = Gamma(α, one(T), check_args = check_args)
function Gamma(α::T, θ::T; check_args=true) where {T <: RTR}
    check_args && Distributions.@check_args(Gamma, α > zero(α) && θ > zero(θ))
    return Gamma{T}(α, θ)
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

function LinearAlgebra.cholesky(A::RTM; check=true)
    factors, info = turing_chol(A, check)
    return Cholesky{eltype(factors), typeof(factors)}(factors, 'U', info)
end

function turing_chol(x::ReverseDiff.TrackedArray{V,D}, check) where {V,D}
    tp = ReverseDiff.tape(x)
    x_value = ReverseDiff.value(x)
    check_value = ReverseDiff.value(check)
    C, back = pullback(_turing_chol, x_value, check_value)
    out = ReverseDiff.track(C.factors, D, tp)
    ReverseDiff.record!(tp, ReverseDiff.SpecialInstruction, turing_chol, (x, check), out, (back, issuccess(C)))
    return out, C.info
end

@noinline function ReverseDiff.special_reverse_exec!(instruction::ReverseDiff.SpecialInstruction{typeof(turing_chol)})
    output = instruction.output
    instruction.cache[2] || throw(PosDefException(C.info))
    input = instruction.input
    input_deriv = ReverseDiff.deriv(input[1])
    P = instruction.cache[1]
    input_deriv .+= P((factors = ReverseDiff.deriv(output),))[1]
    ReverseDiff.unseed!(output)
    return nothing
end

@noinline function special_forward_exec!(instruction::ReverseDiff.SpecialInstruction{typeof(turing_chol)})
    output, input = instruction.output, instruction.input
    C = cholesky(ReverseDiff.value(input[1]), check = ReverseDiff.value(input[2]))
    ReverseDiff.value!(output, C.factors)
    return nothing
end

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
