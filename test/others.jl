using StatsBase: entropy

@testset "unsafe_cholesky" begin
    A = rand(3, 3); A = A + A' + 3I
    @test Matrix(DistributionsAD.unsafe_cholesky(A, true)) == Matrix(cholesky(A))
    @test !issuccess(DistributionsAD.unsafe_cholesky(rand(3,3), false))
    @test_throws PosDefException DistributionsAD.unsafe_cholesky(rand(3,3), true)
end

@testset "TuringWishart" begin
    dim = 3
    A = Matrix{Float64}(I, dim, dim)
    dW1 = Wishart(dim + 4, A)
    dW2 = TuringWishart(dim + 4, A)

    @testset "$F" for F in (size, rank, mean, meanlogdet, entropy, cov, var)
        @test F(dW1) == F(dW2)
    end
    @test Matrix(mode(dW1)) == mode(dW2)
    xw = rand(dW2)
    @test insupport(dW1, xw)
    @test insupport(dW2, xw)
    @test logpdf(dW1, xw) == logpdf(dW2, xw)
end

@testset "TuringInverseWishart" begin
    dim = 3
    A = Matrix{Float64}(I, dim, dim)
    dIW1 = InverseWishart(dim + 4, A)
    dIW2 = TuringInverseWishart(dim + 4, A)

    @testset "$F" for F in (size, rank, mean, cov, var)
        @test F(dIW1) == F(dIW2)
    end
    @test Matrix(mode(dIW1)) == mode(dIW2)
    xiw = rand(dIW2)
    @test insupport(dIW1, xiw)
    @test insupport(dIW2, xiw)
    @test logpdf(dIW1, xiw) == logpdf(dIW2, xiw)
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

    @test entropy(d1) == entropy(d2)
end

@testset "Params" begin
    m = rand(10)
    sigmas = randexp(10)
    
    d = TuringDiagMvNormal(m, sigmas)
    @test params(d) == (m, sigmas)

    d = TuringScalMvNormal(m, sigmas[1])
    @test params(d) == (m, sigmas[1])
end