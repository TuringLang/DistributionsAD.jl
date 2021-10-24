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
    Zygote.z2d(::NTuple{<:Any,Nothing}, ::Tuple) = NoTangent()
    function Zygote.z2d(t::NamedTuple, primal::T) where T
        fnames = fieldnames(T)
        complete_t = map(n -> get(t, n, nothing), fnames)
        primals = map(n -> getfield(primal, n), fnames)
        tp = map(Zygote.z2d, complete_t, primals)
        return if tp isa NTuple{<:Any,NoTangent}
            NoTangent()
        else
            canonicalize(Tangent{T, typeof(tp)}(tp))
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
struct DistSpec{VF<:VariateForm,VS<:ValueSupport,F,T,X,G,B<:Tuple}
    name::Symbol
    f::F
    "Distribution parameters."
    θ::T
    "Sample."
    x::X
    "Transformation of sample `x`."
    xtrans::G
    "Broken backends"
    broken::B
end

function DistSpec(f, θ, x, xtrans=nothing; broken=())
    name = f isa Distribution ? nameof(typeof(f)) : nameof(typeof(f(θ...)))
    return DistSpec(name, f, θ, x, xtrans; broken=broken)
end

function DistSpec(name::Symbol, f, θ, x, xtrans=nothing; broken=())
    F = f isa Distribution ? typeof(f) : typeof(f(θ...))
    VF = Distributions.variate_form(F)
    VS = Distributions.value_support(F)
    return DistSpec{VF,VS,typeof(f),typeof(θ),typeof(x),typeof(xtrans),typeof(broken)}(
        name, f, θ, x, xtrans, broken,
    )
end

Distributions.variate_form(::Type{<:DistSpec{VF}}) where VF = VF
Distributions.value_support(::Type{<:DistSpec{VF,VS}}) where {VF,VS} = VS

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

# Run AD tests of a
function test_ad(dist::DistSpec; kwargs...)
    @info "Testing: $(dist.name)"

    f = dist.f
    θ = dist.θ
    x = dist.x
    g = dist.xtrans
    broken = dist.broken

    # combine all arguments
    # point `x` is not differentiable if the distribution is discrete
    args = if Distributions.value_support(typeof(dist)) === Continuous
        (x, θ...)
    else
        θ
    end

    # Create functions with all arguments
    if Distributions.value_support(typeof(dist)) === Continuous
        f_loglik_allargs = let f=f, g=g
            function (x, θ...)
                dist = f(θ...)
                xtilde = g === nothing ? x : g(x)
                return loglikelihood(dist, xtilde)
            end
        end
        f_logpdf_allargs = let f=f, g=g
            function (x, θ...)
                dist = f(θ...)
                xtilde = g === nothing ? x : g(x)
                if dist isa UnivariateDistribution && xtilde isa AbstractArray
                    return sum(logpdf.(dist, xtilde))
                else
                    return sum(logpdf(dist, xtilde))
                end
            end
        end
    else
        gx = g === nothing ? x : g(x)
        f_loglik_allargs = let f=f, gx=gx
            function (θ...)
                dist = f(θ...)
                return loglikelihood(dist, gx)
            end
        end
        f_logpdf_allargs = let f=f, gx=gx
            function (θ...)
                dist = f(θ...)
                return if dist isa UnivariateDistribution && gx isa AbstractArray
                    sum(logpdf.(dist, gx))
                else
                    sum(logpdf(dist, gx))
                end
            end
        end
    end

    # short cut: since Zygote does not use special number types with
    # different dispatches etc., it is suffiient to just test derivatives of
    # all differentiable arguments at once
    if GROUP === "All" || GROUP === "Zygote"
        @test f_loglik_allargs(args...) ≈ f_logpdf_allargs(args...)

        # Zygote has type inference problems so we don't check it
        try
            for f in (f_loglik_allargs, f_logpdf_allargs)
                test_rrule(
                    Zygote.ZygoteRuleConfig(), f ⊢ NoTangent(), args...;
                    rrule_f=rrule_via_ad, check_inferred=false, kwargs...
                )
            end
        catch
            :Zygote in test_broken || rethrow()
        end
    end

    # early exit
    GROUP !== "Zygote" || return 

    # For all combinations of arguments
    for inds in powerset(1:length(args))
        if !isempty(inds)
            argstest = mapreduce(vcat, inds) do i
                vectorize(args[i])
            end
            f_loglik_test = let args=args, inds=inds
                x -> f_loglik_allargs(unpack(x, inds, args...)...)
            end
            f_logpdf_test = let args=args, inds=inds
                x -> f_logpdf_allargs(unpack(x, inds, args...)...)
            end

            @test f_loglik_test(argstest) ≈ f_logpdf_test(argstest)

            test_ad(f_loglik_test, argstest, broken; kwargs...)
            test_ad(f_logpdf_test, argstest, broken; kwargs...)
        end
    end
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
