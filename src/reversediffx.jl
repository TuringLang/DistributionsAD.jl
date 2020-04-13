# A lot of this module is adapted from Tracker.jl and ReverseDiff.jl
# ReverseDiff.jl is not actively developed but it would be nice to move the code in this 
# module to ReverseDiff at some point.

##########
## fill ##
##########

function Base.fill(
    value::TrackedReal,
    dims::Vararg{Union{Integer, AbstractUnitRange}},
)
    return track(fill, value, dims...)
end
@grad function fill(v::Real, dims...)
    return fill(value(v), dims...), function(Δ)
        size(Δ) ≢  dims && error("Dimension mismatch")
        return (sum(Δ), map(_->nothing, dims)...)
    end
end

###############
## any & all ##
###############

Base.any(f::Function, x::TrackedArray; dims=:) = any(f, value(x), dims = dims)
Base.all(f::Function, x::TrackedArray; dims=:) = all(f, value(x), dims = dims)

#########
## cat ##
#########

function combinations(xs, n)
    n < 1 && return [[]]
    cs = combinations(xs, n-1)
    [[x, c...] for x in xs, c in cs]
end

for f in [:hcat, :vcat]
    for i = 0:2, c = combinations([:AbstractArray, :TrackedArray, :Number, :TrackedReal], i)
        cnames = map(_ -> gensym(), c)
        @eval Base.$f($([:($x::$c) for (x, c) in zip(cnames, c)]...), x::Union{TrackedArray,TrackedReal}, xs::Union{AbstractArray,Number}...) = track($f, $(cnames...), x, xs...)
    end
    @eval begin
        Base.$f(x::TrackedVecOrMat{T}, xs::AbstractVecOrMat{T}...) where T = track($f, x, xs...)
        Base.$f(x1::TrackedVecOrMat{T}, x2::TrackedVecOrMat{T}, xs::AbstractVecOrMat{T}...) where T = track($f, x1, x2, xs...)
        Base.$f(x::TrackedVector{T}, xs::AbstractVector{T}...) where T = track($f, x, xs...)
        Base.$f(x1::TrackedVector{T}, x2::TrackedVector{T}, xs::AbstractVector{T}...) where T = track($f, x1, x2, xs...)

        @grad function $f(x::Real)
            $f(value(x)), (Δ) -> (Δ[1],)
        end
        @grad function $f(x1::Real, x2::Real)
            $f(value(x1), value(x2)), (Δ) -> (Δ[1], Δ[2])
        end
        @grad function $f(x1::AbstractVector, x2::Real)
            $f(value(x1), value(x2)), (Δ) -> (Δ[1:length(x1)], Δ[length(x1)+1])
        end
    end
end

@grad function vcat(xs::Union{TrackedVector, TrackedMatrix}...)
    xs_value = value.(xs)
    out_value = vcat(xs_value...)
    function back(Δ)
        start = 0
        Δs = map(xs) do xsi
          x = map(_ -> :, size(xsi))
          i = isempty(x) ? x : Base.tail(x)
          d = Δ[start+1:start+size(xsi,1), i...]
          start += size(xsi, 1)
          d
        end
        return (Δs...,)
    end
    return out_value, back
end

@grad function hcat(xs::Union{TrackedVector, TrackedMatrix}...)
    xs_value = value.(xs)
    out_value = hcat(xs_value...)
    function back(Δ)
        start = 0
        Δs = map(xs) do xsi
          d = if ndims(xsi) == 1
            Δ[:, start+1]
          else
            i = map(_ -> :, size(xsi)) |> Base.tail |> Base.tail
            Δ[:, start+1:start+size(xsi,2), i...]
          end
          start += size(xsi, 2)
          d
        end
        return (Δs...,)
    end        
    return out_value, back
end

Base.cat(Xs::TrackedArray...; dims) = track(cat, Xs...; dims = dims)
@grad function cat(Xs::TrackedArray{<:Any, D}...; dims) where {D}
    Xs_value = value.(Xs)
    return cat(Xs_value...; dims = dims), Δ -> begin
        start = ntuple(i -> 0, Val(ndims(Δ)))
        Δs = map(Xs) do xs
          dim_xs = 1:ndims(xs)
          till_xs = ntuple((i -> i in dims ? (i in dim_xs ? size(xs,i) : 1) : 0), Val(ndims(Δ)))
          xs_in_Δ = ntuple(i -> till_xs[i] > 0 ? (start[i]+1:start[i]+till_xs[i]) : Colon(), Val(ndims(Δ)))
          d = reshape(Δ[xs_in_Δ...],size(xs))
          start = start .+ till_xs
          d
        end
        return (Δs...,)
    end
end

###############
## logsumexp ##
###############

logsumexp(x::TrackedArray; dims=:) = track(logsumexp, x, dims = dims)
@grad function logsumexp(x::TrackedArray; dims)
    lse = logsumexp(value(x), dims = dims)
    return lse, Δ -> (Δ .* exp.(x .- lse), nothing)
end

############
## linalg ##
############

Base.:*(A::Adjoint{<:Real, <:TrackedVector{<:Real}}, B::AbstractVector{<:Real}) = dot(A, B)
Base.:*(A::Adjoint{<:Real, <:TrackedVector{<:Real}}, B::TrackedVector{<:Real}) = dot(A, B)
Base.:*(A::AbstractVector{<:Real}, B::Adjoint{<:Real, <:TrackedVector{<:Real}}) = dot(A, B)
Base.:*(A::TrackedVector{<:Real}, B::Adjoint{<:Real, <:TrackedVector{<:Real}}) = dot(A, B)

function LinearAlgebra.cholesky(A::TrackedMatrix; check=true)
    factors, info = turing_chol(A, check)
    return Cholesky{eltype(factors), typeof(factors)}(factors, 'U', info)
end

function turing_chol(x::TrackedArray{V,D}, check) where {V,D}
    tp = tape(x)
    x_value = value(x)
    check_value = value(check)
    C, back = Zygote.pullback(_turing_chol, x_value, check_value)
    out = track(C.factors, D, tp)
    record!(tp, SpecialInstruction, turing_chol, (x, check), out, (back, issuccess(C)))
    return out, C.info
end

@noinline function ReverseDiff.special_reverse_exec!(instruction::SpecialInstruction{typeof(turing_chol)})
    output = instruction.output
    instruction.cache[2] || throw(PosDefException(C.info))
    input = instruction.input
    input_deriv = deriv(input[1])
    P = instruction.cache[1]
    input_deriv .+= P((factors = deriv(output),))[1]
    unseed!(output)
    return nothing
end

@noinline function ReverseDiff.special_forward_exec!(instruction::SpecialInstruction{typeof(turing_chol)})
    output, input = instruction.output, instruction.input
    C = cholesky(value(input[1]), check = value(input[2]))
    value!(output, C.factors)
    return nothing
end

##################
## Broadcasting ##
##################

"""
    NotTracked(f::Function)

A struct that can be used to wrap around closures, structs and arrays of structs declaring that they do not contain tracked variables. This enables a more efficient broadcasting of such functions and structs when doing automatic differentiation with `ReverseDiff` producing a `TrackedArray` instead of an `Array{<:TrackedReal}`.
"""
struct NotTracked{F} <: Function
    f::F
end
(f::NotTracked{<:Union{Function, Type}})(args...; kwargs...) = f.f(args...; kwargs...)

istypeorclosure(::F) where {F} = _istypeorclosure(F)
istypeorclosure(::AbstractArray{F}) where {F} = _istypeorclosure(F)
istypeorclosure(::Base.RefValue{F}) where {F} = _istypeorclosure(F)
istypeorclosure(::AbstractArray{<:Real}) = false
istypeorclosure(::TrackedArray) = false
istypeorclosure(::AbstractArray{<:TrackedReal}) = true
istypeorclosure(::Real) = false
@generated _istypeorclosure(::Type{F}) where {F} = :($(fieldcount(F) > 0))

mayhavetracked(b) = istypeorclosure(b)
mayhavetracked(b::Type) = false
mayhavetracked(b::NotTracked) = false
mayhavetracked(b::Base.RefValue{<:NotTracked}) = false
mayhavetracked(b::Broadcasted) = mayhavetracked(b.f) || any(mayhavetracked, b.args)

struct TrackedStyle <: BroadcastStyle end

Broadcast.BroadcastStyle(::Type{<:Union{TrackedArray, TrackedReal}}) = TrackedStyle()
Broadcast.BroadcastStyle(::TrackedStyle, b::BroadcastStyle) = TrackedStyle()

# We have to re-build the original broadcast struct to get the appropriate array
# style. We need this primarily to support CuArrays' broadcasting fixes.
broadcast_rebuild(xs) = value(xs)
function broadcast_rebuild(bc::Broadcasted)
    broadcasted(bc.f, broadcast_rebuild.(bc.args)...)
end

getstyle(::Broadcasted{Style}) where {Style} = Style
remove_not_tracked(f) = f
remove_not_tracked(f::NotTracked) = f.f
remove_not_tracked(f::Base.RefValue{<:NotTracked}) = Ref(remove_not_tracked(f[]))
remove_not_tracked(f::Base.RefValue{<:NotTracked{<:AbstractArray}}) = remove_not_tracked(f[])
function remove_not_tracked(b::Broadcasted{style}) where {style}
    return Broadcasted{style}(remove_not_tracked(b.f), remove_not_tracked.(b.args), b.axes)
end

onlyrealarrays(args::Tuple) = onlyrealarray(first(args)) && onlyrealarrays(Base.tail(args))
onlyrealarrays(::Tuple{}) = true
onlyrealarray(::AbstractArray{<:Real}) = true
onlyrealarray(::AbstractArray) = false
onlyrealarray(::Any) = true

anyreals(args::Tuple) = first(args) isa Real || anyreals(Base.tail(args))
anyreals(args::Tuple{}) = false

function get_implementation(bc, f, T, args)
    outputisreal = (T <: AbstractArray{<:Real}) && (T !== Union{})
    # Each arg is either a real number, an array of untraked reals, a tracked array of reals or an array of untracked non-reals,
    # Output is real, and
    # No tracked closure or arguments, except TrackedReal and TrackedArray.
    if !mayhavetracked(bc) && outputisreal && (anyreals(args) || !onlyrealarrays(args))
        return Val(:tracker)
    # No arg is a real number and array args must be arrays of untracked reals or tracked arrays of reals,
    # Output is real, and
    # No tracked closure or arguments, except TrackedReal and TrackedArray.
    elseif !mayhavetracked(bc) && outputisreal
        return Val(:reversediff)
    # Function or any arg is possibly a tracked non-real or an array of tracked reals/non-reals,
    # Or output is not an array of reals
    else
        return Val(:fallback)
    end
end
function Base.copy(_bc::Broadcasted{TrackedStyle})
    bc = remove_not_tracked(_bc)
    flattened_bc = Broadcast.flatten(bc)
    untracked_bc = broadcast_rebuild(bc)
    flattened_untracked_bc = Broadcast.flatten(untracked_bc)
    T = Core.Compiler.return_type(copy, Tuple{typeof(untracked_bc)})
    f, args = flattened_untracked_bc.f, flattened_bc.args
    implementation = get_implementation(_bc, f, T, args)
    if implementation isa Val{:reversediff}
        return ∇broadcast(f, args...)
    elseif implementation isa Val{:tracker}
        return tracker_∇broadcast(f, args...)
    else
        style, axes = getstyle(flattened_untracked_bc), flattened_bc.axes
        return copy(Broadcasted{style, typeof(axes), typeof(f), typeof(args)}(f, args, axes))
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

getouttype(::TrackedReal{<:Any, D}) where {D} = D
getouttype(::TrackedArray{<:Any, D}) where {D} = D
getouttype(::Any) = Union{}

deref(x) = x
deref(x::Base.RefValue) = x[]

@generated function splatcall(f, x::SVector{N}, utargs::T, ::Val{tinds}) where {N, T <: Tuple, tinds}
    args = []
    ti = 1
    uti = 1
    for i in 1:(N + length(T.types))
        if i in tinds
            push!(args, :(deref(x[$ti])))
            ti += 1
        else
            push!(args, :(deref(utargs[$uti])))
            uti += 1
        end
    end
    return quote
        $(Expr(:meta, :inline))
        $(Expr(:call, :f, args...))
    end
end

@generated function splitargs(args::T) where {T <: Tuple}
    N = length(T.types)
    RealOrArray = Union{Real, AbstractArray}
    inds = [i for i in 1:N if T.types[i] <: RealOrArray]
    indsval = :(Val{$(Expr(:tuple, [:($i) for i in inds]...))}())
    maybetracked = Expr(:tuple, [:(args[$i]) for i in inds]...)
    untracked = Expr(:tuple, [:(args[$i]) for i in 1:N if !(i in inds)]...)
    return :($indsval, $maybetracked, $untracked)
end

## A generalization of the broadcasting approach in ReverseDiff for general functions

@inline function ∇broadcast(f::F, args::Vararg{<:Any}) where {F}
    inds, targs, untracked = splitargs(args)
    N = length(targs)
    D = promote_type(getouttype.(targs)...)
    result = DiffResults.GradientResult(zero(SVector{N, D}))
    function df(x...)
        return ForwardDiff.gradient!(
            result,
            s -> splatcall(f, s, untracked, inds),
            SVector(x),
        )
    end
    results = broadcast(df, value.(targs)...)
    tp = tape(targs...)
    out_value = DiffResults.value.(results)
    eltype(out_value) == Bool && return out_value
    out = track(out_value, D, tp)
	cache = (results, df, ReverseDiff.index_bound.(targs, (out,)))
	record!(tp, SpecialInstruction, ∇broadcast, targs, out, cache)
    return out
end
@noinline function ReverseDiff.special_reverse_exec!(instruction::SpecialInstruction{typeof(∇broadcast)})
    input = instruction.input
    output = instruction.output
    output_deriv = deriv(output)
    results, _, bounds = instruction.cache
    N = length(input)
    if N == 1 || all(isequal(size(input[1])), size.(Base.tail(input)))
        add_to_deriv!(input, output_deriv, results)
    else
        add_to_deriv!(input, output_deriv, results, bounds)
    end
    unseed!(output)
    return nothing
end

@generated function add_to_deriv!(xs::T, o, r) where {T <: Tuple}
    N = length(T.types)
    return Expr(:block, [:(_add_to_deriv!(xs[$i], o, r, Val($i))) for i in 1:N]...)
end
_add_to_deriv!(_, _, _, _) = nothing
function _add_to_deriv!(x::Union{TrackedReal, TrackedArray}, out_deriv, results, ::Val{i}) where {i}
    return ReverseDiff.istracked(x) && ReverseDiff.diffresult_increment_deriv!(x, out_deriv, results, i)
end

@generated function add_to_deriv!(xs::T, o, r, bounds) where {T <: Tuple}
    N = length(T.types)
    return Expr(:block, [:(_add_to_deriv!(xs[$i], o, r, Val($i), bounds[$i])) for i in 1:N]...)
end
_add_to_deriv!(_, _, _, _, _) = nothing
function _add_to_deriv!(x::Union{TrackedReal,TrackedArray}, out_deriv, results, ::Val{i}, bound) where {i}
    return ReverseDiff.istracked(x) && ReverseDiff.diffresult_increment_deriv!(x, out_deriv, results, i, bound)
end

add_to_deriv!(d1, d2) = nothing
function add_to_deriv!(d1::Union{TrackedReal,TrackedArray}, d2)
    ReverseDiff.increment_deriv!(d1, d2)
end

@noinline function ReverseDiff.special_forward_exec!(instruction::SpecialInstruction{typeof(∇broadcast)})
    input, output = instruction.input, instruction.output
    results, df, _ = instruction.cache
    broadcast!(df, results, value.(input)...)
    output_value = value(output)
    output_value .= DiffResults.value.(results)
    return nothing
end

## Tracker style broadcasting
## Good for broadcasting real numbers or arrays of non-tracked structs

trim(x, Δ) = reshape(Δ, ntuple(i -> size(Δ, i), Val(ndims(x))))

unbroadcast(x::AbstractArray, Δ) =
  size(x) == size(Δ) ? Δ :
  length(x) == length(Δ) ? trim(x, Δ) :
    trim(x, sum(Δ, dims = ntuple(i -> size(x, i) == 1 ? i : ndims(Δ)+1, Val(ndims(Δ)))))

unbroadcast(x::Number, Δ) = sum(Δ)
unbroadcast(x::Base.RefValue, _) = nothing

dual(x, p) = x
dual(x::Real, p) = Dual(x, p)

function _deriv(f, G, ::Val{i}, args::Vararg{Any, N}) where {N, i}
    dargs = ntuple(j -> dual(args[j], i==j), Val(N))
    return f(dargs...).partials[1] * G
end
@generated function _derivs(f, G, args::Vararg{Any, N}) where {N}
    return Expr(:tuple, [:(_deriv.(f, G, Val($i), args...)) for i in 1:N]...)
end
@inline function tracker_∇broadcast(f, args::Vararg{Any, N}) where {N}
    args_values = map(value, args)
    out_value = broadcast(f, args_values...)
    tp = tape(args...)
    eltype(out_value) == Bool && return out_value
	out = track(out_value, tp)
    cache = (f,)
	record!(tp, SpecialInstruction, tracker_∇broadcast, args, out, cache)
    return out
end

@noinline function ReverseDiff.special_forward_exec!(instruction::SpecialInstruction{typeof(tracker_∇broadcast)})
    input, output = instruction.input, instruction.output
    f = instruction.cache[1]
    output_value = value(output)
    broadcast!(f, output_value, value.(input)...)
    return nothing
end

@noinline function ReverseDiff.special_reverse_exec!(instruction::SpecialInstruction{typeof(tracker_∇broadcast)})
    input = instruction.input
    output = instruction.output
    f = instruction.cache[1]
    output_deriv = deriv(output)
    N = length(input)
    Δargs = _derivs(f, output_deriv, value.(input)...)
    dxs = map(unbroadcast, input, Δargs)
    map(add_to_deriv!, input, dxs)
    unseed!(output)
    return nothing
end

## Limited ReverseDiff broadcasting
## Efficient broadcasting for specific functions, e.g. +, -

@inline _materialize(f, args) = broadcast(f, args...)

for (M, f, arity) in ReverseDiff.DiffRules.diffrules()
    if arity == 1
        @eval @inline materialize(bc::RDBroadcasted{typeof($M.$f), <:Tuple{TrackedArray}}) = _materialize(bc.f, bc.args)
    elseif arity == 2
        @eval begin
            @inline materialize(bc::RDBroadcasted{typeof($M.$f), <:Tuple{TrackedArray,TrackedArray}}) = _materialize(bc.f, bc.args)
            @inline materialize(bc::RDBroadcasted{typeof($M.$f), <:Tuple{TrackedArray,TrackedReal}}) = _materialize(bc.f, bc.args)
            @noinline materialize(bc::RDBroadcasted{typeof($M.$f), <:Tuple{TrackedReal,TrackedArray}}) = _materialize(bc.f, bc.args)
        end
        for A in ReverseDiff.ARRAY_TYPES
            @eval begin
                @inline materialize(bc::RDBroadcasted{typeof($M.$f), <:Tuple{$A,TrackedArray}}) = _materialize(bc.f, bc.args)
                @inline materialize(bc::RDBroadcasted{typeof($M.$f), <:Tuple{TrackedArray, $A}}) = _materialize(bc.f, bc.args)
                @inline materialize(bc::RDBroadcasted{typeof($M.$f), <:Tuple{$A, TrackedReal}}) = _materialize(bc.f, bc.args)
                @inline materialize(bc::RDBroadcasted{typeof($M.$f), <:Tuple{TrackedReal,$A}}) = _materialize(bc.f, bc.args)
            end
        end
        for R in ReverseDiff.REAL_TYPES
            @eval begin
                @inline materialize(bc::RDBroadcasted{typeof($M.$f), <:Tuple{$R,TrackedArray}}) = _materialize(bc.f, bc.args)
                @inline materialize(bc::RDBroadcasted{typeof($M.$f), <:Tuple{TrackedArray,$R}}) = _materialize(bc.f, bc.args)
            end
        end
    end
end
