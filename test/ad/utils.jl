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

# Auxiliary methods for vectorizing parameters and samples and unflattening them
# similar to `FDM.to_vec`
# However, some implementations in FDM don't work with overload AD such as Tracker,
# ForwardDiff, and ReverseDiff
# Therefore we add a `_to_vec` function

function _to_vec(x::Real)
    function Real_from_vec(v)
        length(v) == 1 || error("vector has incorrect number of elements")
        return first(v)
    end
    return [x], Real_from_vec
end

function _to_vec(x::AbstractArray{<:Real})
    sz = size(x)
    Array_from_vec(v) = reshape(v, sz)
    return vec(x), Array_from_vec
end

function _to_vec(x::Union{Tuple,AbstractVector{<:AbstractArray}})
    x_vecs_and_backs = map(_to_vec, x)
    x_vecs, x_backs = map(first, x_vecs_and_backs), map(last, x_vecs_and_backs)
    lengths = map(length, x_vecs)
    sz = typeof(lengths)(cumsum(collect(lengths)))
    function Tuple_or_Array_of_Array_from_vec(v)
        map(x_backs, lengths, sz) do x_back, l, s
            return x_back(v[(s - l + 1):s])
        end
    end
    return reduce(vcat, x_vecs), Tuple_or_Array_of_Array_from_vec
end

# Functor that fixes non-differentiable location `x` for discrete distributions
struct FixedLocation{X}
    x::X
end
(f::FixedLocation)(args...) = f.x, args

# Functor that transforms differentiable location `x` for continuous distributions
# from unconstrained to constrained space
struct TransformedLocation{F}
    trans::F
end
(f::TransformedLocation)(x, args...) = f.trans(x), args
(f::TransformedLocation{Nothing})(x, args...) = x, args

# Convenience function that returns the correct functor for
# discrete and continuous distributions
make_unpack_x_θ(_, x, ::Type{<:DiscreteDistribution}) = FixedLocation(x)
make_unpack_x_θ(trans, _, ::Type{<:ContinuousDistribution}) = TransformedLocation(trans)

# "Unignore" arguments, i.e., add default arguments if they were ignored
struct Unignore{A}
    args::A
    ignores::BitVector
end

function Unignore(args, ignores::BitVector)
    n = length(args)
    @assert length(ignores) == n
    return Unignore{typeof(args)}(args, ignores)
end

function (f::Unignore)(x...)
    j = Ref(0)
    newx = map(f.args, f.ignores) do argsi, ignoresi
        return if ignoresi
            argsi
        else
            x[(j[] += 1)]
        end
    end

    @assert length(x) == j[] || error("wrong number of arguments")

    return newx
end

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
    broken = dist.broken

    # combine all arguments
    # point `x` is not differentiable if the distribution is discrete
    args = D <: ContinuousDistribution ? (x, θ...) : θ

    # Create function that splits arguments and transforms location x if needed
    unpack_x_θ = make_unpack_x_θ(dist.xtrans, x, D)

    # short cut: since Zygote does not use special number types with
    # different dispatches etc., it is suffiient to just test derivatives of
    # all differentiable arguments at once
    if GROUP === "All" || GROUP === "Zygote"
        # is Zygote broken?
        zygote_broken = :Zygote in broken

        if zygote_broken
            testset_zygote_broken(dist, unpack_x_θ, args...; kwargs...)
        else
            testset_zygote(dist, unpack_x_θ, args...; kwargs...)
        end
    end

    # Early exit
    GROUP !== "Zygote" || return

    # Define functions for computing the log-likelihood that ignore some arguments
    # (i.e., set them to their default values)
    # This is used to check if we can differentiate with respect to a subset of arguments
    # with ForwardDiff, Tracker, and ReverseDiff
    n = length(args)
    ignores = falses(n)
    unignore = Unignore(args, ignores)
    function loglikelihood_test(x...)
        return sum_logpdf_parameterized(unpack_x_θ, f, unignore(x...)...)
    end
    sum_logpdf_test(x...) = sum_logpdf_parameterized(unpack_x_θ, f, unignore(x...)...)

    # Quick sanity check
    @test loglikelihood_test(args...) ≈ sum_logpdf_test(args...)

    # For all combinations of arguments
    for inds in powerset(1:n, 1, n)
        # Update boolean vector of ignored arguments
        fill!(ignores, true)
        for i in inds
            @inbounds ignores[i] = false
        end

        # Vectorize to-be-differentiated arguments for ForwardDiff, Tracker, and ReverseDiff
        args_vec, args_unflatten = _to_vec(args[inds])
        loglik_test(x) = loglikelihood_test(args_unflatten(x)...)
        logpdf_test(x) = sum_logpdf_test(args_unflatten(x)...)

        @test loglik_test(args_vec) ≈ logpdf_test(args_vec)

        test_ad(loglik_test, args_vec, broken; kwargs...)
        test_ad(logpdf_test, args_vec, broken; kwargs...)
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

function testset_zygote(distspec, unpack_x_θ, args...; kwargs...)
    f = distspec.f
    θ = distspec.θ
    x = distspec.x

    @testset "Zygote: $(f(θ...)) at x=$x" begin
        @test loglikelihood_parameterized(unpack_x_θ, f, args...) ≈
            sum_logpdf_parameterized(unpack_x_θ, f, args...)

        for l in (loglikelihood_parameterized, sum_logpdf_parameterized)
            # Zygote has type inference problems so we don't check it
            test_rrule(
                Zygote.ZygoteRuleConfig(), l, unpack_x_θ, f, args...;
                rrule_f=rrule_via_ad, check_inferred=false, kwargs...
            )
        end
    end
end

function testset_zygote_broken(args...; kwargs...)
    # don't show test errors - tests are known to be broken :)
    testset = suppress_stdout() do
        testset_zygote(args...; kwargs...)
    end

    # change errors and fails to broken results, and count number of errors and fails
    efs = errors_to_broken!(testset)

    # ensure that passing tests are not marked as broken
    if iszero(efs)
        error("Zygote tests of $(f(θ...)) at x=$x passed unexpectedly, please mark not as broken")
    end

    return testset
end

# `redirect_stdout(f, devnull)` is only available in Julia >= 1.6
function suppress_stdout(f)
    @static if VERSION < v"1.6"
        open((@static Sys.iswindows() ? "NUL" : "/dev/null"), "w") do devnull
            redirect_stdout(f, devnull)
        end
    else
        redirect_stdout(f, devnull)
    end
end

# change test errors and failures to broken results
function errors_to_broken!(ts::Test.DefaultTestSet)
    results = ts.results
    efs = 0
    for i in eachindex(results)
        @inbounds t = results[i]
        if t isa Test.DefaultTestSet
            efs += errors_to_broken!(t)
        elseif t isa Union{Test.Fail, Test.Error}
            efs += 1
            results[i] = Test.Broken(t.test_type, t.orig_expr)
        end
    end
    return efs
end
