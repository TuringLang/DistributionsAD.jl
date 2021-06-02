@testset "chainrules" begin
    x = randn()
    z = x + exp(randn())
    y = z + exp(randn())
    test_frule(DistributionsAD.uniformlogpdf, x, y, z)
    test_rrule(DistributionsAD.uniformlogpdf, x, y, z)

    # StatsFuns: https://github.com/JuliaStats/StatsFuns.jl/pull/106
    x = exp(randn())
    y = exp(randn())
    z = logistic(randn())
    test_frule(StatsFuns.betalogpdf, x, y, z)
    test_rrule(StatsFuns.betalogpdf, x, y, z)

    x = exp(randn())
    y = exp(randn())
    z = exp(randn())
    test_frule(StatsFuns.gammalogpdf, x, y, z)
    test_rrule(StatsFuns.gammalogpdf, x, y, z)

    x = exp(randn())
    y = exp(randn())
    test_frule(StatsFuns.chisqlogpdf, x, y)
    test_rrule(StatsFuns.chisqlogpdf, x, y)

    x = exp(randn())
    y = exp(randn())
    z = exp(randn())
    test_frule(StatsFuns.fdistlogpdf, x, y, z)
    test_rrule(StatsFuns.fdistlogpdf, x, y, z)

    x = exp(randn())
    y = randn()
    test_frule(StatsFuns.tdistlogpdf, x, y)
    test_rrule(StatsFuns.tdistlogpdf, x, y)

    # TODO: Re-enable if https://github.com/JuliaMath/SpecialFunctions.jl/pull/325 is fixed
    # use `BigFloat` to avoid Rmath implementation in finite differencing check
    # (returns `NaN` for non-integer values)
    #n = rand(1:100)
    #x = BigFloat(n)
    #y = big(logistic(randn()))
    #z = BigFloat(rand(1:n))
    #test_frule(StatsFuns.binomlogpdf, x, y, z)
    #test_rrule(StatsFuns.binomlogpdf, x, y, z)

    x = big(exp(randn()))
    y = BigFloat(rand(1:100))
    test_frule(StatsFuns.poislogpdf, x, y)
    test_rrule(StatsFuns.poislogpdf, x, y)

    _, pb = rrule(StatsFuns.poislogpdf, 0.0, 0.0)
    _, x̄1, _ = pb(1)
    @test x̄1 == -1
    _, pb = rrule(StatsFuns.poislogpdf, 0.0, 1.0)
    _, x̄1, _ = pb(1)
    @test x̄1 == Inf

    # PoissonBinomial
    test_rrule(Distributions.poissonbinomial_pdf_fft, rand(50))
    if isdefined(Distributions, :poissonbinomial_pdf)
        test_rrule(Distributions.poissonbinomial_pdf, rand(50))
    end
end
