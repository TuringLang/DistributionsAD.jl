@testset "others" begin
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
        @testset for TD in (TuringDenseMvNormal, TuringDiagMvNormal, TuringScalMvNormal), T in (Float64, Float32)
            m = rand(T, 3)
            if TD <: TuringDenseMvNormal
                A = rand(T, 3, 3)
                C = A' * A + I
                d1 = TuringMvNormal(m, C)
                d2 = MvNormal(m, C)
            elseif TD <: TuringDiagMvNormal
                C = rand(T, 3)
                d1 = TuringMvNormal(m, C)
                d2 = MvNormal(m, Diagonal(C .^ 2))
            else
                C = rand(T)
                d1 = TuringMvNormal(m, C)
                d2 = MvNormal(m, C^2 * I)
            end

            @testset "$F" for F in (length, size, mean)
                @test F(d1) == F(d2)
            end
            C1 = @inferred(cov(d1))
            @test C1 isa AbstractMatrix{T}
            @test C1 ≈ cov(d2)
            V1 = @inferred(var(d1))
            @test V1 isa AbstractVector{T}
            @test V1 ≈ var(d2)

            x1 = rand(d1)
            x2 = rand(d1, 3)
            for S in (Float64, Float32)
                ST = promote_type(S, T)

                z = map(S, x1)
                logp = @inferred(logpdf(d1, z))
                @test logp isa ST
                @test logp ≈ logpdf(d2, z) rtol = 1e-6

                zs = map(S, x2)
                logps = @inferred(logpdf(d1, zs))
                @test eltype(logps) === ST
                @test logps ≈ logpdf(d2, zs) rtol = 1e-6
            end
        end
    end

    @testset "TuringMvLogNormal" begin
        @testset "$TD" for TD in [TuringDenseMvNormal, TuringDiagMvNormal, TuringScalMvNormal]
            m = rand(3)
            if TD <: TuringDenseMvNormal
                C = Matrix{Float64}(I, 3, 3)
                d1 = TuringMvLogNormal(TuringMvNormal(m, C))
                d2 = MvLogNormal(MvNormal(m, C))
            elseif TD <: TuringDiagMvNormal
                C = ones(3)
                d1 = TuringMvLogNormal(TuringMvNormal(m, C))
                d2 = MvLogNormal(MvNormal(m, Diagonal(C .^ 2)))
            else
                C = 1.0
                d1 = TuringMvLogNormal(TuringMvNormal(m, C))
                d2 = MvLogNormal(MvNormal(m, C^2 * I))
            end

            @test length(d1) == length(d2)

            x1 = rand(d1)
            x2 = rand(d1, 3)
            @test logpdf(d1, x1) ≈ logpdf(d2, x1) rtol=1e-6
            @test logpdf(d1, x2) ≈ logpdf(d2, x2) rtol=1e-6

            x2[:, 1] .= -1
            @test isinf(logpdf(d1, x2)[1])
            @test isinf(logpdf(d2, x2)[1])
        end
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

    @testset "Cholesky" begin
        A = rand(3, 3)'; A = A + A' + 3I;
        C = cholesky(A; check = true)
        factors, info = DistributionsAD.turing_chol(A, true)
        @test factors == C.factors
        @test info == C.info
    end

    @testset "zygote_ldiv" begin
        A = to_posdef(rand(3, 3))
        B = copy(A)
        @test DistributionsAD.zygote_ldiv(A, B) == A \ B
    end

    @testset "Entropy" begin
        sigma = exp(randn())
        d1 = TuringScalMvNormal(randn(10), sigma)
        d2 = MvNormal(randn(10), sigma^2 * I)
        @test entropy(d1) ≈ entropy(d2) rtol=1e-6

        sigmas = exp.(randn(10))
        d1 = TuringDiagMvNormal(randn(10), sigmas)
        d2 = MvNormal(randn(10), Diagonal(sigmas .^ 2))
        @test entropy(d1) ≈ entropy(d2) rtol=1e-6

        A = randn(10)
        C = A * A' + I
        d1 = TuringDenseMvNormal(randn(10), C)
        d2 = MvNormal(randn(10), C)
        @test entropy(d1) ≈ entropy(d2) rtol=1e-6
    end

    @testset "Params" begin
        m = rand(10)
        sigmas = randexp(10)

        d = TuringDiagMvNormal(m, sigmas)
        @test params(d) == (m, sigmas)

        d = TuringScalMvNormal(m, sigmas[1])
        @test params(d) == (m, sigmas[1])
    end

    @testset "adapt_randn" begin
        rng = MersenneTwister()
        for T in (Float32, Float64)
            test_adapt_randn(rng, rand(rng, T, 50), T, 10, 30)
        end
    end

    @testset "TuringDirichlet" begin
        dim = 3
        n = 4
        for alpha in (2, rand())
            d1 = TuringDirichlet(dim, alpha)
            d2 = Dirichlet(dim, alpha)
            d3 = TuringDirichlet(d2)
            @test d1.alpha == d2.alpha == d3.alpha
            @test d1.alpha0 == d2.alpha0 == d3.alpha0
            @test d1.lmnB == d2.lmnB == d3.lmnB

            s1 = rand(d1)
            @test s1 isa Vector{Float64}
            @test length(s1) == dim

            s2 = rand(d1, n)
            @test s2 isa Matrix{Float64}
            @test size(s2) == (dim, n)
        end

        for alpha in (ones(Int, dim), rand(dim))
            d1 = TuringDirichlet(alpha)
            d2 = Dirichlet(alpha)
            d3 = TuringDirichlet(d2)
            @test d1.alpha == d2.alpha == d3.alpha
            @test d1.alpha0 == d2.alpha0 == d3.alpha0
            @test d1.lmnB == d2.lmnB == d3.lmnB

            s1 = rand(d1)
            @test s1 isa Vector{Float64}
            @test length(s1) == dim

            s2 = rand(d1, n)
            @test s2 isa Matrix{Float64}
            @test size(s2) == (dim, n)
        end

        # https://github.com/TuringLang/DistributionsAD.jl/issues/158
        let
            d = TuringDirichlet(rand(2))
            z = rand(d)
            logpdf_z = logpdf(d, z)
            pdf_z = pdf(d, z)

            for x in ([0.5, 0.8], [-0.5, 1.5])
                @test logpdf(d, x) == -Inf
                @test iszero(pdf(d, x))

                xmat = hcat(x, x)
                @test all(==(-Inf), logpdf(d, xmat))
                @test all(iszero, pdf(d, xmat))

                xzmat = hcat(x, z)
                @test logpdf(d, xzmat) == [-Inf, logpdf_z]
                @test pdf(d, xzmat) == [0, pdf_z]
            end
        end
    end
end
