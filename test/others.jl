using StatsBase: entropy

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

@testset "Extras from StatsBase.jl" begin
    sigmas = exp.(randn(10))
    d1 = TuringDiagMvNormal(zeros(10), sigmas)
    d2 = MvNormal(zeros(10), sigmas)

    @test entropy(d1) == entropy(d2)
end
