@testset "AD: Others" begin
    if GROUP == "All" || GROUP == "Tracker"
        @testset "TuringUniform" begin
            @test logpdf(TuringUniform(), param(0.5)) == 0
        end

        @testset "Semicircle" begin
            @test Tracker.data(logpdf(Semicircle(1.0), param(0.5))) == logpdf(Semicircle(1.0), 0.5)
        end
    end

    @testset "logsumexp" begin
        x = rand(3)
        test_reverse_mode_ad(logsumexp, randn(), x; rtol=1e-8, atol=1e-6)
    end

    @testset "zygote_ldiv" begin
        A = to_posdef(rand(3, 3))
        B = to_posdef(rand(3, 3))
        
        test_reverse_mode_ad(randn(3, 3), A, B) do A, B
            return DistributionsAD.zygote_ldiv(A, B)
        end
    end

    @testset "logdet" begin
        N = 7
        B = randn(N, N)

        test_reverse_mode_ad(randn(), B; rtol=1e-8, atol=1e-6) do B
            return logdet(cholesky(to_posdef(B)))
        end
        test_reverse_mode_ad(randn(), B; rtol=1e-8, atol=1e-6) do B
            return logdet(cholesky(Symmetric(to_posdef(B))))
        end
    end

    @testset "fill" begin
        if GROUP == "All" || GROUP == "Tracker"
            @test fill(param(1.0), 3) isa TrackedArray
        end

        test_reverse_mode_ad(x->fill(x, 7), randn(7), randn())
        test_reverse_mode_ad(x->fill(x, 7, 11), randn(7, 11), randn())
        test_reverse_mode_ad(x->fill(x, 7, 11, 13), rand(7, 11, 13), randn())
    end

    @testset "Tracker, Zygote and ReverseDiff + MvNormal" begin
        N = 7
        m = rand(N)
        B = randn(N, N)
        x = rand(TuringDenseMvNormal(m, to_posdef(B)))

        test_reverse_mode_ad(randn(), m, B, x) do m, B, x
            return logpdf(MvNormal(m, to_posdef(B)), x)
        end
        test_reverse_mode_ad(randn(), m, B, x) do m, B, x
            return logpdf(TuringMvNormal(m, to_posdef(B)), x)
        end
        test_reverse_mode_ad(randn(), m, B, x) do m, B, x
            return logpdf(TuringMvNormal(m, Symmetric(to_posdef(B))), x)
        end
    end

    @testset "adapt_randn" begin
        rng = MersenneTwister()
        n = 50
        dims = (10, 30)
        for T in (Float32, Float64)
            if GROUP == "All" || GROUP == "ForwardDiff"
                let
                    x = [ForwardDiff.Dual(rand(rng, T)) for _ in 1:n]
                    test_adapt_randn(rng, x, T, dims...)
                end
            end
            if GROUP == "All" || GROUP == "Tracker"
                let
                    x = Tracker.TrackedArray(rand(rng, T, 50))
                    test_adapt_randn(rng, x, T, dims...)
                end
            end
            if GROUP == "All" || GROUP == "ReverseDiff"
                let
                    v = rand(rng, T, n)
                    d = rand(Int, n)
                    tp = ReverseDiff.InstructionTape()
                    x = ReverseDiff.TrackedArray(v, d, tp) 
                    test_adapt_randn(rng, x, T, dims...)
                end
            end
        end
    end
end