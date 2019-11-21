struct ADTestFunction
    name::String
    f::Function
    x::Vector
end
struct DistSpec{Tθ<:Tuple, Tx}
    name::Union{Symbol, Expr}
    θ::Tθ
    x::Tx
end

vectorize(v::Number) = [v]
vectorize(v) = vec(v)
pack(vals...) = reduce(vcat, vectorize.(vals))
function unpack(x, vals...)
    unpacked = []
    i = 1
    for v in vals
        if v isa Number
            push!(unpacked, x[i])
            i += 1
        elseif v isa Vector
            push!(unpacked, x[i:i+length(v)-1])
            i += length(v)
        elseif v isa Matrix
            push!(unpacked, reshape(x[i:(i+length(v)-1)], size(v)))
            i += length(v)
        else
            throw("Unsupported argument")
        end
    end
    @assert i == length(x) + 1
    return (unpacked...,)
end
function get_function(dist::DistSpec, inds, val)
    syms = []
    args = []
    for (i, a) in enumerate(dist.θ)
        if i in inds
            sym = gensym()
            push!(syms, sym)
            push!(args, sym)
        else
            push!(args, a)
        end
    end
    if val
        sym = gensym()
        push!(syms, sym)
        expr = :(($(syms...),) -> logpdf($(dist.name)($(args...)), $(sym)))
        if length(inds) == 0
            f = x -> Base.invokelatest(eval(expr), unpack(x, dist.x)...)
            return ADTestFunction(string(expr), f, pack(dist.x))
        else
            f = x -> Base.invokelatest(eval(expr), unpack(x, dist.θ[inds]..., dist.x)...)
            return ADTestFunction(string(expr), f, pack(dist.θ[inds]..., dist.x))
        end
    else
        @assert length(inds) > 0
        expr = :(($(syms...),) -> logpdf($(dist.name)($(args...)), $(dist.x)))
        f = x -> Base.invokelatest(eval(expr), unpack(x, dist.θ[inds]...)...)
        return ADTestFunction(string(expr), f, pack(dist.θ[inds]...))
    end
end
function get_all_functions(dist::DistSpec, continuous=false)
    fs = []
    if length(dist.θ) == 0
        push!(fs, get_function(dist, (), true))
    else
        for inds in combinations(1:length(dist.θ))
            push!(fs, get_function(dist, inds, false))
            if continuous
                push!(fs, get_function(dist, inds, true))
            end
        end
    end
    return fs
end

function test_ad(f, at = 0.5; rtol = 1e-8, atol = 1e-8)
    isarr = isa(at, AbstractArray)
    reverse = Tracker.data(Tracker.gradient(f, at)[1])
    if isarr
        forward = ForwardDiff.gradient(f, at)
        @test isapprox(reverse, forward, rtol=rtol, atol=atol)
    else
        forward = ForwardDiff.derivative(f, at)
        finite_diff = central_fdm(5,1)(f, at)
        @test isapprox(reverse, forward, rtol=rtol, atol=atol)
        @test isapprox(reverse, finite_diff, rtol=rtol, atol=atol)
    end
end

"""
    test_reverse_mode_ad(forward, f, ȳ, x...; rtol=1e-8, atol=1e-8)

Check that the reverse-mode sensitivities produced by an AD library are correct for `f`
at `x...`, given sensitivity `ȳ` w.r.t. `y = f(x...)` up to `rtol` and `atol`.
`forward` should be either `Tracker.forward` or `Zygote.pullback`.
"""
function test_reverse_mode_ad(forward, f, ȳ, x...; rtol=1e-8, atol=1e-8)

    # Perform a regular forwards-pass.
    y = f(x...)

    # Use tracker to compute reverse-mode sensitivities.
    y_tracker, back = forward(f, x...)
    x̄s_tracker = back(ȳ)

    # Use finite differencing to compute reverse-mode sensitivities.
    x̄s_fdm = FDM.j′vp(central_fdm(5, 1), f, ȳ, x...)
    if length(x) == 1
        x̄s_fdm = (x̄s_fdm,)
    end

    # Check that forwards-pass produces the correct answer.
    @test y ≈ y_tracker

    # Check that reverse-mode sensitivities are correct.
    @test all([x̄_tracker ≈ x̄_fdm for (x̄_tracker, x̄_fdm) in zip(x̄s_tracker, x̄s_fdm)])
end

# See `test_reverse_mode_ad` for details.
function test_tracker_ad(f, ȳ, x...; rtol=1e-8, atol=1e-8)
    return test_reverse_mode_ad(Tracker.forward, f, ȳ, x...; rtol=rtol, atol=atol)
end

_to_cov(B) = B * B' + Matrix(I, size(B)...)
