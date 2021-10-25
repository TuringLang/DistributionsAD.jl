using ChainRulesCore
using ChainRulesTestUtils
using FiniteDifferences

const FDM = FiniteDifferences

# Load AD backends
if GROUP == "All" || GROUP == "ForwardDiff"
    @eval using ForwardDiff
end
if GROUP == "All" || GROUP == "Zygote"
    @eval using Zygote

    # Workaround for nested `nothing`
    # Partly copied from https://github.com/FluxML/Zygote.jl/pull/1104
    # TODO: Remove if https://github.com/FluxML/Zygote.jl/pull/1104 is merged
    Zygote.z2d(::NTuple{<:Any,Nothing}, ::Tuple) = NoTangent()
    function Zygote.z2d(delta::NamedTuple, primal::T) where T
        fnames = fieldnames(T)
        deltas = map(n -> get(delta, n, nothing), fnames)
        primals = map(n -> getfield(primal, n), fnames)
        inner = map(Zygote.z2d, deltas, primals)
        return if inner isa Tuple{Vararg{AbstractZero}}
            NoTangent()
        else
            backing = NamedTuple{fnames}(inner)
            canonicalize(Tangent{T, typeof(backing)}(backing))
        end
    end
end
if GROUP == "All" || GROUP == "ReverseDiff"
    @eval using ReverseDiff
end
if GROUP == "All" || GROUP == "Tracker"
    @eval using Tracker
end

function test_reverse_mode_ad(f, ȳ, x...; rtol=1e-6, atol=1e-6)
    # Perform a regular forwards-pass.
    y = f(x...)

    # Use finite differencing to compute reverse-mode sensitivities.
    x̄s_fdm = FDM.j′vp(central_fdm(5, 1), f, ȳ, x...)

    if GROUP == "All" || GROUP == "Zygote"
        # Use Zygote to compute reverse-mode sensitivities.
        y_zygote, back_zygote = Zygote.pullback(f, x...)
        x̄s_zygote = back_zygote(ȳ)

        # Check that Zygpte forwards-pass produces the correct answer.
        @test y ≈ y_zygote atol=atol rtol=rtol

        # Check that Zygote reverse-mode sensitivities are correct.
        @test all(zip(x̄s_zygote, x̄s_fdm)) do (x̄_zygote, x̄_fdm)
            return isapprox(x̄_zygote, x̄_fdm; atol=atol, rtol=rtol)
        end
    end

    if GROUP == "All" || GROUP == "ReverseDiff"
        test_rd = length(x) == 1 && y isa Number
        if test_rd
            # Use ReverseDiff to compute reverse-mode sensitivities.
            if x[1] isa Array
                x̄s_rd = similar(x[1])
                tp = ReverseDiff.GradientTape(x -> f(x), x[1])
                ReverseDiff.gradient!(x̄s_rd, tp, x[1])
                x̄s_rd .*= ȳ
                y_rd = ReverseDiff.value(tp.output)
                @assert y_rd isa Number
            else
                x̄s_rd = [x[1]]
                tp = ReverseDiff.GradientTape(x -> f(x[1]), [x[1]])
                ReverseDiff.gradient!(x̄s_rd, tp, [x[1]])
                y_rd = ReverseDiff.value(tp.output)[1]
                x̄s_rd = x̄s_rd[1] * ȳ
                @assert y_rd isa Number
            end

            # Check that ReverseDiff forwards-pass produces the correct answer.
            @test y ≈ y_rd atol=atol rtol=rtol

            # Check that ReverseDiff reverse-mode sensitivities are correct.
            @test x̄s_rd ≈ x̄s_fdm[1] atol=atol rtol=rtol
        end
    end

    if GROUP == "All" || GROUP == "Tracker"
        # Use Tracker to compute reverse-mode sensitivities.
        y_tracker, back_tracker = Tracker.forward(f, x...)
        x̄s_tracker = back_tracker(ȳ)

        # Check that Tracker forwards-pass produces the correct answer.
        @test y ≈ Tracker.data(y_tracker) atol=atol rtol=rtol

        # Check that Tracker reverse-mode sensitivities are correct.
        @test all(zip(x̄s_tracker, x̄s_fdm)) do (x̄_tracker, x̄_fdm)
            return isapprox(Tracker.data(x̄_tracker), x̄_fdm; atol=atol, rtol=rtol)
        end
    end
end

# Define pullback for `to_simplex`
function to_simplex_pullback(ȳ::AbstractArray, y::AbstractArray)
    x̄ = ȳ .* y
    x̄ .= x̄ .- y .* sum(x̄; dims=1)
    return x̄
end
function ChainRulesCore.rrule(::typeof(to_simplex), x::AbstractArray{<:Real})
    y = to_simplex(x)
    pullback(ȳ) = (NoTangent(), to_simplex_pullback(ȳ, y))
    return y, pullback
end

# Define adjoints for ReverseDiff
if GROUP == "All" || GROUP == "ReverseDiff"
    @eval begin
        function to_simplex(x::AbstractArray{<:ReverseDiff.TrackedReal})
            return ReverseDiff.track(to_simplex, x)
        end
        ReverseDiff.@grad function to_simplex(x)
            _x = ReverseDiff.value(x)
            y = to_simplex(_x)
            pullback(ȳ) = (to_simplex_pullback(ȳ, y),)
            return y, pullback
        end
    end
end

# Define adjoints for Tracker
if GROUP == "All" || GROUP == "Tracker"
    @eval begin
        to_posdef(A::Tracker.TrackedMatrix) = Tracker.track(to_posdef, A)
        Tracker.@grad function to_posdef(A::Tracker.TrackedMatrix)
            data_A = Tracker.data(A)
            S = data_A * data_A' + I
            function pullback(∇)
                return ((∇ + ∇') * data_A,)
            end
            return S, pullback
        end

        to_simplex(x::Tracker.TrackedArray) = Tracker.track(to_simplex, x)
        Tracker.@grad function to_simplex(x::Tracker.TrackedArray)
            data_x = Tracker.data(x)
            y = to_simplex(data_x)
            pullback(ȳ) = (to_simplex_pullback(ȳ, y),)
            return y, pullback
        end
    end
end

# Struct of distribution, corresponding parameters, and a sample.
struct DistSpec{D,F,T,X,G,B<:Tuple}
    f::F
    "Distribution parameters."
    θ::T
    "Sample."
    x::X
    "Transformation of sample `x`."
    xtrans::G
    "Broken backends"
    broken::B

    function DistSpec{D}(f::F, θ, x, xtrans::T, broken) where {D,F,T}
        return new{D,F,typeof(θ),typeof(x),T,typeof(broken)}(f, θ, x, xtrans, broken)
    end
end

function DistSpec(d::Distribution, θ, x, xtrans=nothing; broken=())
    return DistSpec{typeof(d)}(d, θ, x, xtrans, broken)
end

function DistSpec(f::F, θ, x, xtrans=nothing; broken=()) where {F}
    D = typeof(f(θ...))
    return DistSpec{D}(f, θ, x, xtrans, broken)
end

dist_type(::DistSpec{D}) where {D} = D

# Auxiliary method for vectorizing parameters and samples
vectorize(v::Number) = [v]
vectorize(v::Diagonal) = v.diag
vectorize(v::AbstractVector{<:AbstractMatrix}) = mapreduce(vectorize, vcat, v)
vectorize(v) = vec(v)

"""
    unpack(x, inds, original...)

Return a tuple of unpacked parameters and samples in vector `x`.

Here `original` are the original full set of parameters and samples, and
`inds` contains the indices of the original parameters and samples for which
a possibly different value is given in `x`. If no value is provided in `x`,
the original value of the parameter is returned. The values are returned
in the same order as the original parameters.
"""
function unpack(x, inds, original...)
    offset = 0
    newvals = ntuple(length(original)) do i
        if i in inds
            v, offset = unpack_offset(x, offset, original[i])
        else
            v = original[i]
        end
        return v
    end
    offset == length(x) || throw(ArgumentError())

    return newvals
end

# Auxiliary methods for unpacking numbers and arrays
function unpack_offset(x, offset, original::Number)
    newoffset = offset + 1
    val = x[newoffset]
    return val, newoffset
end
function unpack_offset(x, offset, original::AbstractArray)
    newoffset = offset + length(original)
    val = reshape(x[(offset + 1):newoffset], size(original))
    return val, newoffset
end
function unpack_offset(x, offset, original::AbstractArray{<:AbstractArray})
    newoffset = offset
    val = map(original) do orig
        out, newoffset = unpack_offset(x, newoffset, orig)
        return out
    end
    return val, newoffset
end

# functor that fixes non-differentiable location `x` for discrete distributions
struct FixedLocation{X}
    x::X
end
(f::FixedLocation)(args...) = f.x, args

# functor that transforms differentiable location `x` for continuous distributions
# from unconstrained to constrained space
struct TransformedLocation{F}
    trans::F
end
(f::TransformedLocation)(x, args...) = f.trans(x), args
(f::TransformedLocation{Nothing})(x, args...) = x, args

# convenience function that returns the correct functor for
# discrete and continuous distributions
make_unpack_x_θ(_, x, ::Type{<:DiscreteDistribution}) = FixedLocation(x)
make_unpack_x_θ(trans, _, ::Type{<:ContinuousDistribution}) = TransformedLocation(trans)

# we define the following two functions to be able to tell Zygote that it should not
# compute derivatives for the fields of the functors `unpack_x_θ`
"""
    loglikelihood_parameterized(unpack_x_θ, dist, args...)

Compute the log-likelihood of distribution `dist(θ...)` for `x` where
`x, θ = unpack_x_θ(args...)` are extracted from the arguments `args` with `unpack_x_θ`.

Internally, computations are performed with `loglikelihood`.

See also: [`sum_logpdf_parameterized`](@ref)
"""
function loglikelihood_parameterized(unpack_x_θ, f, args...)
    x, θ = ignore_derivatives(unpack_x_θ)(args...)
    return loglikelihood(f(θ...), x)
end

"""
    sum_logpdf_parameterized(unpack_x_θ, dist, args...)

Compute the log-likelihood of distribution `dist(θ...)` for `x` where
`x, θ = unpack_x_θ(args...)` are extracted from the arguments `args` with `unpack_x_θ`.

Internally, the log pdf of individual data points is computed with `logpdf` which are then
summed up.

See also: [`loglikelihood_parameterized`](@ref)
"""
function sum_logpdf_parameterized(unpack_x_θ, f, args...)
    x, θ = ignore_derivatives(unpack_x_θ)(args...)
    # we use `_logpdf` to be able to handle univariate distributions correctly (see below)
    return sum(_logpdf(f(θ...), x))
end

# Function that computes arrays of `logpdf` values
# `logpdf` does not handle arrays of samples for univariate distributions
_logpdf(d::Distribution, x) = logpdf(d, x)
_logpdf(d::UnivariateDistribution, x::AbstractArray) = logpdf.((d,), x)

# Run AD tests
function test_ad(dist::DistSpec{D}; kwargs...) where {D}
    f = dist.f
    θ = dist.θ
    x = dist.x
    g = dist.xtrans
    broken = dist.broken

    # combine all arguments
    # point `x` is not differentiable if the distribution is discrete
    args = D <: ContinuousDistribution ? (x, θ...) : θ

    # Create function that splits arguments and transforms location x if needed
    unpack_x_θ = make_unpack_x_θ(g, x, D)

    # short cut: since Zygote does not use special number types with
    # different dispatches etc., it is suffiient to just test derivatives of
    # all differentiable arguments at once
    if GROUP === "All" || GROUP === "Zygote"
        @test loglikelihood_parameterized(unpack_x_θ, f, args...) ≈
            sum_logpdf_parameterized(unpack_x_θ, f, args...)

        # Zygote has type inference problems so we don't check it
        try
            for l in (loglikelihood_parameterized, sum_logpdf_parameterized)
                test_rrule(
                    Zygote.ZygoteRuleConfig(), l, unpack_x_θ, f, args...;
                    rrule_f=rrule_via_ad, check_inferred=false, kwargs...
                )
            end
        catch
            :Zygote in broken || rethrow()
        end
    end

    # Early exit
    GROUP !== "Zygote" || return

    # For all combinations of arguments
    for inds in powerset(1:length(args))
        if !isempty(inds)
            argstest = mapreduce(vcat, inds) do i
                vectorize(args[i])
            end

            # Make functions with vectorized to-be-differentiated arguments for ForwardDiff, Tracker, and ReverseDiff
            loglikelihood_test = let l=loglikelihood_parameterized, g=unpack_x_θ, f=f, args=args, inds=inds
                x -> l(g, f, unpack(x, inds, args...)...)
            end
            sum_logpdf_test = let l=sum_logpdf_parameterized, g=unpack_x_θ, f=f, args=args, inds=inds
                x -> l(g, f, unpack(x, inds, args...)...)
            end

            @test loglikelihood_test(argstest) ≈ sum_logpdf_test(argstest)

            test_ad(loglikelihood_test, argstest, broken; kwargs...)
            test_ad(sum_logpdf_test, argstest, broken; kwargs...)
        end
    end

    return
end

function test_ad(f, x, broken = (); rtol = 1e-6, atol = 1e-6)
    finitediff = FDM.grad(central_fdm(5, 1), f, x)[1]

    if GROUP == "All" || GROUP == "Tracker"
        if :Tracker in broken
            @test_broken Tracker.data(Tracker.gradient(f, x)[1]) ≈ finitediff rtol=rtol atol=atol
        else
            @test Tracker.data(Tracker.gradient(f, x)[1]) ≈ finitediff rtol=rtol atol=atol
        end
    end

    if GROUP == "All" || GROUP == "ForwardDiff"
        if :ForwardDiff in broken
            @test_broken ForwardDiff.gradient(f, x) ≈ finitediff rtol=rtol atol=atol
        else
            @test ForwardDiff.gradient(f, x) ≈ finitediff rtol=rtol atol=atol
        end
    end

    if GROUP == "All" || GROUP == "ReverseDiff"
        if :ReverseDiff in broken
            @test_broken ReverseDiff.gradient(f, x) ≈ finitediff rtol=rtol atol=atol
        else
            @test ReverseDiff.gradient(f, x) ≈ finitediff rtol=rtol atol=atol
        end
    end

    return
end
