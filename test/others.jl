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
        @testset "$TD" for TD in [TuringDenseMvNormal, TuringDiagMvNormal, TuringScalMvNormal]
            m = rand(3)
            if TD <: TuringDenseMvNormal
                A = rand(3, 3)
                C = A' * A + I
                d1 = TuringMvNormal(m, C)
            elseif TD <: TuringDiagMvNormal
                C = rand(3)
                d1 = TuringMvNormal(m, C)
            else
                C = rand()
                d1 = TuringMvNormal(m, C)
            end
            d2 = MvNormal(m, C)

            @testset "$F" for F in (length, size, mean)
                @test F(d1) == F(d2)
            end
            @test cov(d1) ≈ cov(d2)
            @test var(d1) ≈ var(d2)

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
        @test logpdf(TuringUniform(), 0.5) == 0
        if AD == "All" || AD == "Tracker"
            @test logpdf(TuringUniform(), param(0.5)) == 0
        end
    end

    if AD == "All" || AD == "Tracker"
        @testset "Semicircle" begin
            @test Tracker.data(logpdf(Semicircle(1.0), param(0.5))) == logpdf(Semicircle(1.0), 0.5)
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

    function test_reverse_mode_ad( f, ȳ, x...; rtol=1e-6, atol=1e-6)
        # Perform a regular forwards-pass.
        y = f(x...)

        # Use finite differencing to compute reverse-mode sensitivities.
        x̄s_fdm = FDM.j′vp(central_fdm(5, 1), f, ȳ, x...)

        if AD == "All" || AD == "Zygote"
            # Use Zygote to compute reverse-mode sensitivities.
            y_zygote, back_zygote = Zygote.pullback(f, x...)
            x̄s_zygote = back_zygote(ȳ)

            # Check that Zygpte forwards-pass produces the correct answer.
            @test isapprox(y, y_zygote, atol=atol, rtol=rtol)

            # Check that Zygote reverse-mode sensitivities are correct.
            @test all(zip(x̄s_zygote, x̄s_fdm)) do (x̄_zygote, x̄_fdm)
                isapprox(x̄_zygote, x̄_fdm; atol=atol, rtol=rtol)
            end
        end

        if AD == "All" || AD == "ReverseDiff"
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
                @test isapprox(y, y_rd, atol=atol, rtol=rtol)

                # Check that ReverseDiff reverse-mode sensitivities are correct.
                @test isapprox(x̄s_rd, x̄s_fdm[1]; atol=atol, rtol=rtol)
            end
        end

        if AD == "All" || AD == "Tracker"
            # Use Tracker to compute reverse-mode sensitivities.
            y_tracker, back_tracker = Tracker.forward(f, x...)
            x̄s_tracker = back_tracker(ȳ)

            # Check that Tracker forwards-pass produces the correct answer.
            @test isapprox(y, Tracker.data(y_tracker), atol=atol, rtol=rtol)

            # Check that Tracker reverse-mode sensitivities are correct.
            @test all(zip(x̄s_tracker, x̄s_fdm)) do (x̄_tracker, x̄_fdm)
                isapprox(Tracker.data(x̄_tracker), x̄_fdm; atol=atol, rtol=rtol)
            end
        end
    end
    _to_cov(B) = B + B' + 2 * size(B, 1) * Matrix(I, size(B)...)

    @testset "logsumexp" begin
        x, y = rand(3), rand()
        test_reverse_mode_ad(logsumexp, y, x; rtol=1e-8, atol=1e-6)
    end

    @testset "zygote_ldiv" begin
        A = rand(3, 3)'; A = A + A' + 3I;
        B = copy(A)
        Ȳ = rand(3, 3)
        @test DistributionsAD.zygote_ldiv(A, B) == A \ B
        test_reverse_mode_ad((A,B)->DistributionsAD.zygote_ldiv(A,B), Ȳ, A, B)
    end

    @testset "logdet" begin
        rng, N = MersenneTwister(123456), 7
        y, B = randn(rng), randn(rng, N, N)
        test_reverse_mode_ad(B->logdet(cholesky(_to_cov(B))), y, B; rtol=1e-8, atol=1e-6)
        test_reverse_mode_ad(B->logdet(cholesky(Symmetric(_to_cov(B)))), y, B; rtol=1e-8, atol=1e-6)
    end

    @testset "fill" begin
        if AD == "All" || AD == "Tracker"
            @test fill(param(1.0), 3) isa TrackedArray
        end
        rng = MersenneTwister(123456)
        test_reverse_mode_ad(x->fill(x, 7), randn(rng, 7), randn(rng))
        test_reverse_mode_ad(x->fill(x, 7, 11), randn(rng, 7, 11), randn(rng))
        test_reverse_mode_ad(x->fill(x, 7, 11, 13), rand(rng, 7, 11, 13), randn(rng))
    end
    @testset "Tracker, Zygote and ReverseDiff + MvNormal" begin
        rng, N = MersenneTwister(123456), 11
        B = randn(rng, N, N)
        m, A = randn(rng, N), B' * B + I

        # Generate from the TuringDenseMvNormal
        d = TuringDenseMvNormal(m, A)
        x = rand(d)

        # Check that the logpdf agrees with MvNormal.
        d_ref = MvNormal(m, PDMat(A))
        @test logpdf(d, x) ≈ logpdf(d_ref, x)

        test_reverse_mode_ad((m, B, x)->logpdf(MvNormal(m, _to_cov(B)), x), randn(rng), m, B, x)
        test_reverse_mode_ad((m, B, x)->logpdf(TuringMvNormal(m, _to_cov(B)), x), randn(rng), m, B, x)
        test_reverse_mode_ad((m, B, x)->logpdf(TuringMvNormal(m, Symmetric(_to_cov(B))), x), randn(rng), m, B, x)
    end

    @testset "Entropy" begin
        sigma = exp(randn())
        d1 = TuringScalMvNormal(randn(10), sigma)
        d2 = MvNormal(randn(10), sigma)
        @test entropy(d1) ≈ entropy(d2) rtol=1e-6

        sigmas = exp.(randn(10))
        d1 = TuringDiagMvNormal(randn(10), sigmas)
        d2 = MvNormal(randn(10), sigmas)
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

        xs = Any[(rng, T, n) -> rand(rng, T, n)]
        if AD == "All" || AD == "ForwardDiff"
            push!(xs, (rng, T, n) -> [ForwardDiff.Dual(rand(rng, T)) for _ in 1:n])
        end
        if AD == "All" || AD == "Tracker"
            push!(xs, (rng, T, n) -> Tracker.TrackedArray(rand(rng, T, n)))
        end
        if AD == "All" || AD == "ReverseDiff"
            push!(xs, (rng, T, n) -> begin
                  v = rand(rng, T, n)
                  d = rand(Int, n)
                  tp = ReverseDiff.InstructionTape()
                  ReverseDiff.TrackedArray(v, d, tp)
                  end)
        end

        for T in (Float32, Float64)
            for f in xs
                x = f(rng, T, 50)

                Random.seed!(rng, 100)
                y = DistributionsAD.adapt_randn(rng, x, 10, 30)
                @test y isa Matrix{T}
                @test size(y) == (10, 30)

                Random.seed!(rng, 100)
                @test y == randn(rng, T, 10, 30)
            end
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
