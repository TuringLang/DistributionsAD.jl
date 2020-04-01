const RTR = ReverseDiff.TrackedReal
const RTV = ReverseDiff.TrackedVector
const RTM = ReverseDiff.TrackedMatrix
const RTA = ReverseDiff.TrackedArray
using DistributionsAD.ReverseDiffX: @grad
using StatsBase: entropy

if get_stage() in ("Others", "all")
    @testset "TuringWishart" begin
        dim = 3
        A = Matrix{Float64}(I, dim, dim)
        dW1 = Wishart(dim + 4, A)
        dW2 = TuringWishart(dim + 4, A)
        dW3 = TuringWishart(dW1)
        mean = Distributions.mean
        @testset "$F" for F in (size, rank, mean, meanlogdet, entropy, cov, var)
            @test F(dW1) == F(dW2) == F(dW3)
        end
        @test Matrix(mode(dW1)) == mode(dW2) == mode(dW3)
        xw = rand(dW2)
        @test insupport(dW1, xw)
        @test insupport(dW2, xw)
        @test insupport(dW3, xw)
        @test logpdf(dW1, xw) == logpdf(dW2, xw) == logpdf(dW3, xw)
    end

    @testset "TuringInverseWishart" begin
        dim = 3
        A = Matrix{Float64}(I, dim, dim)
        dIW1 = InverseWishart(dim + 4, A)
        dIW2 = TuringInverseWishart(dim + 4, A)
        dIW3 = TuringInverseWishart(dIW1)
        mean = Distributions.mean
        @testset "$F" for F in (size, rank, mean, cov, var)
            @test F(dIW1) == F(dIW2) == F(dIW3)
        end
        @test Matrix(mode(dIW1)) == mode(dIW2) == mode(dIW3)
        xiw = rand(dIW2)
        @test insupport(dIW1, xiw)
        @test insupport(dIW2, xiw)
        @test insupport(dIW3, xiw)
        @test logpdf(dIW1, xiw) == logpdf(dIW2, xiw) == logpdf(dIW3, xiw)
    end

    @testset "TuringMvNormal" begin
        @testset "$TD" for TD in [TuringDenseMvNormal, TuringDiagMvNormal, TuringScalMvNormal]
            m = rand(3)
            if TD <: TuringDenseMvNormal
                C = Matrix{Float64}(I, 3, 3)
                d1 = TuringMvNormal(m, C)
            elseif TD <: TuringDiagMvNormal
                C = ones(3)
                d1 = TuringMvNormal(m, C)
            else
                C = 1.0
                d1 = TuringMvNormal(m, C)
            end
            d2 = MvNormal(m, C)

            @testset "$F" for F in (length, size)
                @test F(d1) == F(d2)
            end

            x1 = rand(d1)
            x2 = rand(d1, 3)
            @test isapprox(logpdf(d1, x1), logpdf(d2, x1), rtol = 1e-6)
            @test isapprox(logpdf(d1, x2), logpdf(d2, x2), rtol = 1e-6)
        end
    end

    @testset "TuringMvLogNormal" begin
        @testset "$TD" for TD in [TuringDenseMvNormal, TuringDiagMvNormal, TuringScalMvNormal]
            m = rand(3)
            if TD <: TuringDenseMvNormal
                C = Matrix{Float64}(I, 3, 3)
                d1 = TuringMvLogNormal(TuringMvNormal(m, C))
            elseif TD <: TuringDiagMvNormal
                C = ones(3)
                d1 = TuringMvLogNormal(TuringMvNormal(m, C))
            else
                C = 1.0
                d1 = TuringMvLogNormal(TuringMvNormal(m, C))
            end
            d2 = MvLogNormal(MvNormal(m, C))

            @test length(d1) == length(d2)

            x1 = rand(d1)
            x2 = rand(d1, 3)
            @test isapprox(logpdf(d1, x1), logpdf(d2, x1), rtol = 1e-6)
            @test isapprox(logpdf(d1, x2), logpdf(d2, x2), rtol = 1e-6)

            x2[:, 1] .= -1
            @test isinf(logpdf(d1, x2)[1])
            @test isinf(logpdf(d2, x2)[1])
        end
    end

    @testset "TuringUniform" begin
        @test logpdf(TuringUniform(), param(0.5)) == 0
    end

    @testset "Semicircle" begin
        @test Tracker.data(logpdf(Semicircle(1.0), param(0.5))) == logpdf(Semicircle(1.0), 0.5)
    end

    @testset "TuringPoissonBinomial" begin
        d1 = TuringPoissonBinomial([0.5, 0.5])
        d2 = PoissonBinomial([0.5, 0.5])
        @test quantile(d1, 0.5) == quantile(d2, 0.5)
        @test minimum(d1) == minimum(d2)
    end

    @testset "Inverse of pi" begin
        @test 1/pi == inv(pi)
    end

    @testset "Others" begin
        @test fill(param(1.0), 3) isa TrackedArray
        x = rand(3)
        @test isapprox(Tracker.data(Tracker.gradient(logsumexp, x)[1]),
            ForwardDiff.gradient(logsumexp, x), atol = 1e-5)
        A = rand(3, 3)'; A = A + A' + 3I;
        C = cholesky(A; check = true)
        factors, info = DistributionsAD.turing_chol(A, true)
        @test factors == C.factors
        @test info == C.info
        B = copy(A)
        @test DistributionsAD.zygote_ldiv(A, B) == A \ B
    end

    @testset "Entropy" begin
        sigmas = exp.(randn(10))
        d1 = TuringDiagMvNormal(zeros(10), sigmas)
        d2 = MvNormal(zeros(10), sigmas)

        @test isapprox(entropy(d1), entropy(d2), rtol = 1e-6)
    end

    @testset "Params" begin
        m = rand(10)
        sigmas = randexp(10)
        
        d = TuringDiagMvNormal(m, sigmas)
        @test params(d) == (m, sigmas)

        d = TuringScalMvNormal(m, sigmas[1])
        @test params(d) == (m, sigmas[1])
    end

    @testset "ReverseDiff @grad macro" begin
        x = rand(3);
        A = rand(3, 3);
        A_x = [vec(A); x];
        global custom_grad_called
        
        f1(x) = dot(x, x)
        f1(x::RTV) = ReverseDiff.track(f1, x)
        @grad function f1(x::AbstractVector)
            global custom_grad_called = true
            xv = ReverseDiff.value(x)
            dot(xv, xv), Δ -> (Δ * 2 * xv,)
        end
        
        custom_grad_called = false
        g1 = ReverseDiff.gradient(f1, x)
        g2 = ReverseDiff.gradient(x -> dot(x, x), x)
        @test g1 == g2
        @test custom_grad_called
        
        f2(A, x) = A * x
        f2(A, x::RTV) = ReverseDiff.track(f2, A, x)
        f2(A::RTM, x) = ReverseDiff.track(f2, A, x)
        f2(A::RTM, x::RTV) = ReverseDiff.track(f2, A, x)
        @grad function f2(A::AbstractMatrix, x::AbstractVector)
            global custom_grad_called = true
            Av = ReverseDiff.value(A)
            xv = ReverseDiff.value(x)
            Av * xv, Δ -> (Δ * xv', Av' * Δ)
        end
        
        custom_grad_called = false
        g1 = ReverseDiff.gradient(x -> sum(f2(A, x)), x)
        g2 = ReverseDiff.gradient(x -> sum(A * x), x)
        @test g1 == g2
        @test custom_grad_called
        
        custom_grad_called = false
        g1 = ReverseDiff.gradient(A -> sum(f2(A, x)), A)
        g2 = ReverseDiff.gradient(A -> sum(A * x), A)
        @test g1 == g2
        @test custom_grad_called
        
        custom_grad_called = false
        g1 = ReverseDiff.gradient(A_x -> sum(f2(reshape(A_x[1:9], 3, 3), A_x[10:end])), A_x)
        g2 = ReverseDiff.gradient(A_x -> sum(reshape(A_x[1:9], 3, 3) * A_x[10:end]), A_x)
        @test g1 == g2
        @test custom_grad_called
    
        f3(A; dims) = sum(A, dims = dims)
        f3(A::RTM; dims) = ReverseDiff.track(f3, A; dims = dims)
        @grad function f3(A::AbstractMatrix; dims)
            global custom_grad_called = true
            Av = ReverseDiff.value(A)
            sum(Av, dims = dims), Δ -> (zero(Av) .+ Δ,)
        end
        custom_grad_called = false
        g1 = ReverseDiff.gradient(A -> sum(f3(A, dims = 1)), A)
        g2 = ReverseDiff.gradient(A -> sum(sum(A, dims = 1)), A)
        @test g1 == g2
        @test custom_grad_called
    
        f4(::typeof(log), A; dims) = sum(log, A, dims = dims)
        f4(::typeof(log), A::RTM; dims) = ReverseDiff.track(f4, log, A; dims = dims)
        @grad function f4(::typeof(log), A::AbstractMatrix; dims)
            global custom_grad_called = true
            Av = ReverseDiff.value(A)
            sum(log, Av, dims = dims), Δ -> (nothing, 1 ./ Av .* Δ)
        end
        custom_grad_called = false
        g1 = ReverseDiff.gradient(A -> sum(f4(log, A, dims = 1)), A)
        g2 = ReverseDiff.gradient(A -> sum(sum(log, A, dims = 1)), A)
        @test g1 == g2
        @test custom_grad_called
    
        f5(x) = log(x)
        f5(x::RTR) = ReverseDiff.track(f5, x)
        @grad function f5(x::Real)
            global custom_grad_called = true
            xv = ReverseDiff.value(x)
            log(xv), Δ -> (1 / xv * Δ,)
        end
        custom_grad_called = false
        g1 = ReverseDiff.gradient(x -> f5(x[1]) * f5(x[2]) + exp(x[3]), x)
        g2 = ReverseDiff.gradient(x -> log(x[1]) * log(x[2]) + exp(x[3]), x)
        @test g1 == g2
        @test custom_grad_called
    
        f6(x) = sum(x)
        f6(x::RTA{<:AbstractFloat}) = ReverseDiff.track(f6, x)
        @grad function f6(x::RTA{T}) where {T <: AbstractFloat}
            global custom_grad_called = true
            xv = ReverseDiff.value(x)
            sum(xv), Δ -> (one.(xv) .* Δ,)
        end
    
        custom_grad_called = false
        g1 = ReverseDiff.gradient(f6, x)
        g2 = ReverseDiff.gradient(sum, x)
        @test g1 == g2
        @test custom_grad_called
        
        x2 = round.(Int, x)
        custom_grad_called = false
        g1 = ReverseDiff.gradient(f6, x2)
        g2 = ReverseDiff.gradient(sum, x2)
        @test g1 == g2
        @test !custom_grad_called
        f6(x::RTA) = ReverseDiff.track(f6, x)
        @test_throws MethodError ReverseDiff.gradient(f6, x2)

        f7(x...) = +(x...)
        f7(x::RTR{<:AbstractFloat}...) = ReverseDiff.track(f7, x...)
        @grad function f7(x::RTR{T}...) where {T <: AbstractFloat}
            global custom_grad_called = true
            xv = ReverseDiff.value.(x)
            +(xv...), Δ -> one.(xv) .* Δ
        end
        custom_grad_called = false
        g1 = ReverseDiff.gradient(x -> f7(x...), x)
        g2 = ReverseDiff.gradient(sum, x)
        @test g1 == g2
        @test custom_grad_called

        f8(A; kwargs...) = sum(A, kwargs...)
        f8(A::RTM; kwargs...) = ReverseDiff.track(f8, A; kwargs...)
        @grad function f8(A::AbstractMatrix; kwargs...)
            global custom_grad_called = true
            Av = ReverseDiff.value(A)
            sum(Av; kwargs...), Δ -> (zero(Av) .+ Δ,)
        end
        custom_grad_called = false
        g1 = ReverseDiff.gradient(A -> sum(f8(A, dims = 1)), A)
        g2 = ReverseDiff.gradient(A -> sum(sum(A, dims = 1)), A)
        @test g1 == g2
        @test custom_grad_called
    end
end