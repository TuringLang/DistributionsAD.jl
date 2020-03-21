const RTR = ReverseDiff.TrackedReal
const RTV = ReverseDiff.TrackedVector
const RTM = ReverseDiff.TrackedMatrix
const RTA = ReverseDiff.TrackedArray
using ReverseDiff: SpecialInstruction
import NaNMath
using ForwardDiff: Dual
import SpecialFunctions: logbeta
import Distributions: Gamma

Base.:*(A::Adjoint{<:Real, <:RTV{<:Real}}, B::AbstractVector{<:Real}) = dot(A, B)
Base.:*(A::Adjoint{<:Real, <:RTV{<:Real}}, B::RTV{<:Real}) = dot(A, B)
Base.:*(A::AbstractVector{<:Real}, B::Adjoint{<:Real, <:RTV{<:Real}}) = dot(A, B)
Base.:*(A::RTV{<:Real}, B::Adjoint{<:Real, <:RTV{<:Real}}) = dot(A, B)

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
    ReverseDiff.record!(tp, SpecialInstruction, turing_chol, (x, check), out, (back, issuccess(C)))
    return out, C.info
end

@noinline function ReverseDiff.special_reverse_exec!(instruction::SpecialInstruction{typeof(turing_chol)})
    output = instruction.output
    instruction.cache[2] || throw(PosDefException(C.info))
    input = instruction.input
    input_deriv = ReverseDiff.deriv(input[1])
    P = instruction.cache[1]
    input_deriv .+= P((factors = ReverseDiff.deriv(output),))[1]
    ReverseDiff.unseed!(output)
    return nothing
end

@noinline function ReverseDiff.special_forward_exec!(instruction::SpecialInstruction{typeof(turing_chol)})
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

# Modified from Tracker.jl

Base.vcat(xs::RTM...) = _vcat(xs...)
Base.vcat(xs::RTV...) = _vcat(xs...)
function _vcat(xs::Union{RTV{<:Any, D}, RTM{<:Any, D}}...) where {D}
    tp = ReverseDiff.tape(xs...)
    xs_value = ReverseDiff.value.(xs)
    out_value = vcat(xs_value...)
    function back(Δ)
        start = 0
        Δs = [begin
          x = map(_ -> :, size(xsi))
          i = isempty(x) ? x : Base.tail(x)
          d = Δ[start+1:start+size(xsi,1), i...]
          start += size(xsi, 1)
          d
        end for xsi in xs]
        return (Δs...,)
    end        
    out = ReverseDiff.track(out_value, D, tp)
    ReverseDiff.record!(tp, SpecialInstruction, vcat, xs, out, (back,))
    return out
end

@noinline function ReverseDiff.special_reverse_exec!(instruction::SpecialInstruction{typeof(vcat)})
    output = instruction.output
    input = instruction.input
    input_derivs = ReverseDiff.deriv.(input)
    P = instruction.cache[1]
    jtvs = P(ReverseDiff.deriv(output))
    for i in 1:length(input_derivs)
        input_derivs[i] .+= jtvs[i]
    end
    ReverseDiff.unseed!(output)
    return nothing
end

@noinline function ReverseDiff.special_forward_exec!(instruction::SpecialInstruction{typeof(vcat)})
    output, input = instruction.output, instruction.input
    out_value = vcat(ReverseDiff.value.(input)...)
    ReverseDiff.value!(output, out_value)
    return nothing
end

Base.hcat(xs::RTM...) = _hcat(xs...)
Base.hcat(xs::RTV...) = _hcat(xs...)
function _hcat(xs::Union{RTV{<:Any, D}, RTM{<:Any, D}}...) where {D}
    tp = ReverseDiff.tape(xs...)
    xs_value = ReverseDiff.value.(xs)
    out_value = hcat(xs_value...)
    function back(Δ)
        start = 0
        Δs = [begin
          d = if ndims(xsi) == 1
            Δ[:, start+1]
          else
            i = map(_ -> :, size(xsi)) |> Base.tail |> Base.tail
            Δ[:, start+1:start+size(xsi,2), i...]
          end
          start += size(xsi, 2)
          d
        end for xsi in xs]
        return (Δs...,)
    end        
    out = ReverseDiff.track(out_value, D, tp)
    ReverseDiff.record!(tp, SpecialInstruction, hcat, xs, out, (back,))
    return out
end

@noinline function ReverseDiff.special_reverse_exec!(instruction::SpecialInstruction{typeof(hcat)})
    output = instruction.output
    input = instruction.input
    input_derivs = ReverseDiff.deriv.(input)
    P = instruction.cache[1]
    jtvs = P(ReverseDiff.deriv(output))
    for i in 1:length(input_derivs)
        input_derivs[i] .+= jtvs[i]
    end
    ReverseDiff.unseed!(output)
    return nothing
end

@noinline function ReverseDiff.special_forward_exec!(instruction::SpecialInstruction{typeof(hcat)})
    output, input = instruction.output, instruction.input
    out_value = hcat(ReverseDiff.value.(input)...)
    ReverseDiff.value!(output, out_value)
    return nothing
end

Base.cat(Xs::RTA...; dims) = _cat(dims, Xs...)
Base.cat(Xs::RTV...; dims) = _cat(dims, Xs...)
function _cat(dims, Xs::Union{RTV{<:Any, D}, RTM{<:Any, D}}...) where {D}
    tp = ReverseDiff.tape(dims, Xs...)
    Xs_value = ReverseDiff.value.(Xs)
    out_value = cat(Xs_value...; dims = dims)
    function back(Δ)
        start = ntuple(i -> 0, Val(ndims(Δ)))
        Δs = [begin
          dim_xs = 1:ndims(xs)
          till_xs = ntuple((i -> i in dims ? (i in dim_xs ? size(xs,i) : 1) : 0), Val(ndims(Δ)))
          xs_in_Δ = ntuple(i -> till_xs[i] > 0 ? (start[i]+1:start[i]+till_xs[i]) : Colon(), Val(ndims(Δ)))
          d = reshape(Δ[xs_in_Δ...],size(xs))
          start = start .+ till_xs
          d
        end for xs in Xs]
        return (Δs...,)
    end        
    out = ReverseDiff.track(out_value, D, tp)
    ReverseDiff.record!(tp, SpecialInstruction, cat, (dims, Xs...), out, (back,))
    return out
end

@noinline function ReverseDiff.special_reverse_exec!(instruction::SpecialInstruction{typeof(cat)})
    output = instruction.output
    input = instruction.input
    input_derivs = ReverseDiff.deriv.(Base.tail(input))
    P = instruction.cache[1]
    jtvs = P(ReverseDiff.deriv(output))
    for i in 1:length(jtvs)
        input_derivs[i] .+= jtvs[i]
    end
    ReverseDiff.unseed!(output)
    return nothing
end

@noinline function ReverseDiff.special_forward_exec!(instruction::SpecialInstruction{typeof(cat)})
    output, input = instruction.output, instruction.input
    dims = ReverseDiff.value(input[1])
    Xs = ReverseDiff.value.(Base.tail(input))
    out_value = cat(Xs..., dims = dims)
    ReverseDiff.value!(output, out_value)
    return nothing
end

###########

# Broadcasting

using ReverseDiff: ForwardOptimize
using Base.Broadcast: Broadcasted
import Base.Broadcast: materialize
const RDBroadcasted{F, T} = Broadcasted{<:Any, <:Any, F, T}

_materialize(f, args) = broadcast(ForwardOptimize(f), args...)

for (M, f, arity) in ReverseDiff.DiffRules.diffrules()
    if arity == 1
        @eval @inline materialize(bc::RDBroadcasted{typeof($M.$f), <:Tuple{RTA}}) = _materialize(bc.f, bc.args)
    elseif arity == 2
        @eval begin
            @inline materialize(bc::RDBroadcasted{typeof($M.$f), <:Tuple{RTA, RTA}}) = _materialize(bc.f, bc.args)
            @inline materialize(bc::RDBroadcasted{typeof($M.$f), <:Tuple{RTA, RTR}}) = _materialize(bc.f, bc.args)
            @inline materialize(bc::RDBroadcasted{typeof($M.$f), <:Tuple{RTR, RTA}}) = _materialize(bc.f, bc.args)
        end
        for A in ReverseDiff.ARRAY_TYPES
            @eval begin
                @inline materialize(bc::RDBroadcasted{typeof($M.$f), <:Tuple{$A, RTA}}) = _materialize(bc.f, bc.args)
                @inline materialize(bc::RDBroadcasted{typeof($M.$f), <:Tuple{RTA, $A}}) = _materialize(bc.f, bc.args)
                @inline materialize(bc::RDBroadcasted{typeof($M.$f), <:Tuple{$A, RTR}}) = _materialize(bc.f, bc.args)
                @inline materialize(bc::RDBroadcasted{typeof($M.$f), <:Tuple{RTR, $A}}) = _materialize(bc.f, bc.args)
            end
        end
        for R in ReverseDiff.REAL_TYPES
            @eval begin
                @inline materialize(bc::RDBroadcasted{typeof($M.$f), <:Tuple{$R, RTA}}) = _materialize(bc.f, bc.args)
                @inline materialize(bc::RDBroadcasted{typeof($M.$f), <:Tuple{RTA, $R}}) = _materialize(bc.f, bc.args)
            end
        end
    end
end
