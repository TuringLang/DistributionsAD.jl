@testset "chainrules" begin
    x = randn()
    z = x + exp(randn())
    y = z + exp(randn())
    test_frule(DistributionsAD.uniformlogpdf, x, y, z)
    test_rrule(DistributionsAD.uniformlogpdf, x, y, z)

    # Only check the following rules if Distributions does not use CR
    if !isdefined(Distributions, :ChainRulesCore)
    # PoissonBinomial
    test_rrule(Distributions.poissonbinomial_pdf_fft, rand(50))
    if isdefined(Distributions, :poissonbinomial_pdf)
        test_rrule(Distributions.poissonbinomial_pdf, rand(50))
    end
    end
end
