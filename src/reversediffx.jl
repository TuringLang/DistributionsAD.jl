module ReverseDiffX

# A lot of this module is adapted from Tracker.jl.
# ReverseDiff.jl is not actively developed but it would be nice to move the code in this 
# module to ReverseDiff at some point.

export NotTracked

using MacroTools, LinearAlgebra
using ForwardDiff: Dual
import SpecialFunctions, NaNMath, Zygote
using ..ReverseDiff
const RTR = ReverseDiff.TrackedReal
const RTV = ReverseDiff.TrackedVector
const RTM = ReverseDiff.TrackedMatrix
const RTA = ReverseDiff.TrackedArray
using ..ReverseDiff: SpecialInstruction
using ..DistributionsAD: DistributionsAD, _turing_chol
import ..DistributionsAD: turing_chol
using Base.Broadcast: BroadcastStyle, ArrayStyle, Broadcasted, broadcasted

"""
    f(x) = dot(x, x)
    f(x::ReverseDiff.TrackedVector) = ReverseDiff.track(f, x)
    ReverseDiff.@grad function f(x)
        xv = ReverseDiff.value(x)
        return dot(xv, xv), ∇ -> (∇ * 2 * xv,)
    end
The `@grad` macro provides a way for the users to define custom adjoints for single-output functions wrt to their input numbers or arrays.
"""
macro grad(expr)
    if @capture(expr, 
        (f_(xs__) where {T__} = body_) | 
        (f_(xs__) = body_) | 
        (function f_(xs__) body_ end) | 
        (function f_(xs__) where {T__} body_ end)
    )
        closure = gensym(:f)
        tp = gensym(:tp)
        output_value = gensym(:output_value)
        output = gensym(:output)
        back = gensym(:back)
        args = gensym(:args)
        xsv = getargs_expr(xs)
        T = T == nothing ? [] : T
        return quote
            function ReverseDiff.track(::typeof($f), $(xs...)) where {$(T...),}
                $args = $xsv
                $closure = ($(xs...),) -> $body
                $tp = ReverseDiff.tape($args...)
                $output_value, $back = $closure($args...)
                $output = ReverseDiff.track($output_value, $tp)
                ReverseDiff.record!(
                    $tp,
                    ReverseDiff.SpecialInstruction,
                    $f,
                    $args,
                    $output,
                    ($back, $closure),
                )
                return $output
            end

            @static if !hasmethod(
                ReverseDiff.special_reverse_exec!,
                Tuple{ReverseDiff.SpecialInstruction{typeof($f)}},
            )
                @noinline function ReverseDiff.special_reverse_exec!(instruction::ReverseDiff.SpecialInstruction{typeof($f)})
                    output = instruction.output
                    input = instruction.input
                    back = instruction.cache[1]
                    input_derivs = back(ReverseDiff.deriv(output))
                    @assert input_derivs isa Tuple
                    ReverseDiff.add_to_deriv!.(input, input_derivs)
                    ReverseDiff.unseed!(output)
                    return nothing
                end
            end

            @static if !hasmethod(
                ReverseDiff.special_forward_exec!,
                Tuple{ReverseDiff.SpecialInstruction{typeof($f)}},
            )
                @noinline function ReverseDiff.special_forward_exec!(instruction::ReverseDiff.SpecialInstruction{typeof($f)})
                    output, input = instruction.output, instruction.input
                    pullback = instruction.cache[2]
                    out_value = pullback(input...)[1]
                    ReverseDiff.value!(output, out_value)
                    return nothing
                end
            end
        end |> esc
    else
        throw("Invalid `ReverseDiff` custom gradient definition.")
    end
end
add_to_deriv!(d1, d2) = nothing
function add_to_deriv!(d1::Union{RTR, RTA}, d2)
    d = ReverseDiff.deriv(d1)
    d .+= d2
end
function getargs_expr(args_with_types)
    expr = Expr(:tuple)
    for at in args_with_types
        x, tosplat = remove_tp(at)
        if tosplat
            push!(expr.args, :($x...))
        else
            push!(expr.args, x)
        end
    end
    return expr
end
function remove_tp(t)
    if @capture(t, X_::T_...)
        return X, true
    elseif @capture(t, X_::T_)
        return X, false
    elseif @capture(t, ::typeof(T_)...)
        return T, true
    elseif @capture(t, ::typeof(T_))
        return T, false
    elseif @capture(t, X_...)
        return X, true
    else
        return t, false
    end
end

_fill(v::Real, dims::Vararg{Union{Integer, AbstractUnitRange}}) = fill(v[], dims...)
Base.fill(v::RTR, dims::Vararg{Union{Integer, AbstractUnitRange}}) = _fill(Ref(v), dims...)
function _fill(
    value::Base.RefValue{<:RTR},
    dims::Vararg{Union{Integer, AbstractUnitRange}},
)
    return ReverseDiff.track(_fill, value, dims...)
end
@grad function _fill(value::Base.RefValue{<:Real}, dims...)
    return fill(ReverseDiff.value(value[]), dims...), function(Δ)
        size(Δ) ≢  dims && error("Dimension mismatch")
        return (sum(Δ), map(_->nothing, dims)...)
    end
end

Base.:*(A::Adjoint{<:Real, <:RTV{<:Real}}, B::AbstractVector{<:Real}) = dot(A, B)
Base.:*(A::Adjoint{<:Real, <:RTV{<:Real}}, B::RTV{<:Real}) = dot(A, B)
Base.:*(A::AbstractVector{<:Real}, B::Adjoint{<:Real, <:RTV{<:Real}}) = dot(A, B)
Base.:*(A::RTV{<:Real}, B::Adjoint{<:Real, <:RTV{<:Real}}) = dot(A, B)

function LinearAlgebra.cholesky(A::RTM; check=true)
    factors, info = turing_chol(A, check)
    return Cholesky{eltype(factors), typeof(factors)}(factors, 'U', info)
end

function turing_chol(x::ReverseDiff.TrackedArray{V,D}, check) where {V,D}
    tp = ReverseDiff.tape(x)
    x_value = ReverseDiff.value(x)
    check_value = ReverseDiff.value(check)
    C, back = Zygote.pullback(_turing_chol, x_value, check_value)
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

# Modified from Tracker.jl

Base.vcat(xs::RTM...) = ReverseDiff.track(vcat, xs...)
Base.vcat(xs::RTV...) = ReverseDiff.track(vcat, xs...)
@grad function vcat(xs::Union{RTV, RTM}...)
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
    return out_value, back
end

Base.hcat(xs::RTM...) = ReverseDiff.track(hcat, xs...)
Base.hcat(xs::RTV...) = ReverseDiff.track(hcat, xs...)
@grad function hcat(xs::Union{RTV, RTM}...)
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
    return out_value, back
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

using ForwardDiff: Dual, partials

trim(x, Δ) = reshape(Δ, ntuple(i -> size(Δ, i), Val(ndims(x))))

unbroadcast(x::AbstractArray, Δ) =
  	size(x) == size(Δ) ? Δ :
  	length(x) == length(Δ) ? trim(x, Δ) :
    	trim(x, sum(Δ, dims = ntuple(i -> size(x, i) == 1 ? i : ndims(Δ)+1, Val(ndims(Δ)))))

unbroadcast(x::Number, Δ) = sum(Δ)
unbroadcast(x::Base.RefValue, _) = nothing

dual(x, p) = x
dual(x::Real, p) = Dual(x, p)

function partial(f, Δ, i, args::Vararg{Any,N}) where {N}
  dargs = ntuple(j -> dual(args[j], i==j), Val(N))
  return Δ * f(dargs...).partials[1]
end

isclosure(::Any) = false
@generated isclosure(::F) where {F <: Function} = :($(fieldcount(F) > 0))
hasclosure(b) = isclosure(b)
hasclosure(b::Broadcasted) = isclosure(b.f) || any(hasclosure, b.args)

"""
    NotTracked(f::Function)

A callable struct that can be used to wrap around closures declaring that they are not closures of tracked variables. This enables the broadcasting of such functions producing a `TrackedArray` instead of an `Array{<:TrackedReal}`.
"""
struct NotTracked{F <: Function} <: Function
    f::F
end
(f::NotTracked)(args...; kwargs...) = f.f(args...; kwargs...)

@inline maybetrackedclosure(f) = false
@inline maybetrackedclosure(f::NotTracked) = false
@inline maybetrackedclosure(f::Function) = isclosure(f)
@inline mayhavetrackedclosure(b) = false
@inline mayhavetrackedclosure(b::Broadcasted) = maybetrackedclosure(b.f) || 
    any(mayhavetrackedclosure, b.args)

@inline function ∇broadcast(untracked_bc, fallback_style, axes, f::F, args::Vararg{<:Any,N}) where {F, N}
    y = Base.materialize(untracked_bc)
    tp = ReverseDiff.tape(f, args...)
    eltype(y) <: Real || return copy(Broadcasted{fallback_style, typeof(axes), typeof(f), typeof(args)}(f, args, axes))
    eltype(y) == Bool && return y
    function back(Δ)
        Δargs = ntuple(i -> partial.(f, Δ, i, args...), Val(N))
        dxs = map(unbroadcast, args, Δargs)
        return dxs
    end
    out = ReverseDiff.track(y, tp)
    _args = map(args) do a
        a isa Number && return Ref(a)
        return a
    end
    ReverseDiff.record!(tp, ReverseDiff.SpecialInstruction, ∇broadcast, _args, out, (back, untracked_bc))
    return out
end
@noinline function ReverseDiff.special_reverse_exec!(instruction::ReverseDiff.SpecialInstruction{typeof(∇broadcast)})
    output = instruction.output
    input = instruction.input
    back = instruction.cache[1]
    input_derivs = back(ReverseDiff.deriv(output))
    @assert input_derivs isa Tuple
    ReverseDiff.add_to_deriv!.(input, input_derivs)
    ReverseDiff.unseed!(output)
    return nothing
end
@noinline function ReverseDiff.special_forward_exec!(instruction::ReverseDiff.SpecialInstruction{typeof(∇broadcast)})
    output, input = instruction.output, instruction.input
    bc = instruction.cache[2]
    out_value = Base.materialize(bc)
    ReverseDiff.value!(output, out_value)
    return nothing
end

struct TrackedStyle <: BroadcastStyle end

Broadcast.BroadcastStyle(::Type{<:Union{RTA, RTR}}) = TrackedStyle()
Broadcast.BroadcastStyle(::TrackedStyle, b::BroadcastStyle) = TrackedStyle()

# We have to re-build the original broadcast struct to get the appropriate array
# style. We need this primarily to support CuArrays' broadcasting fixes.
broadcast_rebuild(xs) = ReverseDiff.value(xs)
function broadcast_rebuild(bc::Broadcasted)
    broadcasted(bc.f, broadcast_rebuild.(bc.args)...)
end
preprocess(x) = x

getstyle(::Broadcasted{Style}) where {Style} = Style
function Base.copy(bc::Broadcasted{TrackedStyle})
    bc1 = Broadcast.flatten(bc)
    untracked_bc = broadcast_rebuild(bc)
    bc2 = Broadcast.flatten(untracked_bc)
    style = getstyle(bc2)
    axes = bc1.axes
    f, args = bc2.f, bc1.args
    T = Core.Compiler.return_type(f, Tuple{eltype.(args)...})
    maybereal = T <: Real || T >: Real
    if hasclosure(bc) && mayhavetrackedclosure(bc) || !maybereal
        return copy(Broadcasted{style, typeof(axes), typeof(f), typeof(args)}(f, args, axes))
    else
        return ∇broadcast(untracked_bc, style, axes, f, args...)
    end
end

# https://github.com/FluxML/Flux.jl/issues/353
if VERSION < v"1.1.0-DEV.548"
    @eval Base.Broadcast begin
        function flatten(bc::Broadcasted{Style}) where {Style}
            isflat(bc) && return bc
            args = cat_nested(bc)
            let makeargs = make_makeargs(bc), f = bc.f
                newf = @inline function(args::Vararg{Any,N}) where N
                f(makeargs(args...)...)
                end
                return Broadcasted{Style}(newf, args, bc.axes)
            end
        end
        @inline function make_makeargs(makeargs, t::Tuple{<:Broadcasted,Vararg{Any}})
            bc = t[1]
            let makeargs = make_makeargs(makeargs, tail(t)), f = bc.f
                let makeargs = make_makeargs(makeargs, bc.args)
                    headargs, tailargs = make_headargs(bc.args), make_tailargs(bc.args)
                    return @inline function(args::Vararg{Any,N}) where N
                        args1 = makeargs(args...)
                        a, b = headargs(args1...), tailargs(args1...)
                        (f(a...), b...)
                    end
                end
            end
        end
    end
end

end
