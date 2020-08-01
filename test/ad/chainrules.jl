
@testset "chainrules" begin
    x, Δx, x̄ = randn(3)
    y, Δy, ȳ = randn(3)
    z, Δz, z̄ = randn(3)
    Δu = randn()

    ỹ = x + exp(y) + exp(z)
    z̃ = x + exp(y)
    frule_test(DistributionsAD.uniformlogpdf, (x, Δx), (ỹ, Δy), (z̃, Δz))
    rrule_test(DistributionsAD.uniformlogpdf, Δu, (x, x̄), (ỹ, ȳ), (z̃, z̄))

    x̃ = exp(x)
    ỹ = exp(y)
    z̃ = logistic(z)
    frule_test(DistributionsAD.betalogpdf, (x̃, Δx), (ỹ, Δy), (z̃, Δz))
    rrule_test(DistributionsAD.betalogpdf, Δu, (x̃, x̄), (ỹ, ȳ), (z̃, z̄))

    x̃ = exp(x)
    ỹ = exp(y)
    z̃ = exp(z)
    frule_test(DistributionsAD.gammalogpdf, (x̃, Δx), (ỹ, Δy), (z̃, Δz))
    rrule_test(DistributionsAD.gammalogpdf, Δu, (x̃, x̄), (ỹ, ȳ), (z̃, z̄))

    x̃ = exp(x)
    ỹ = exp(y)
    z̃ = exp(z)
    frule_test(DistributionsAD.chisqlogpdf, (x̃, Δx), (ỹ, Δy))
    rrule_test(DistributionsAD.chisqlogpdf, Δu, (x̃, x̄), (ỹ, ȳ))

    x̃ = exp(x)
    ỹ = exp(y)
    z̃ = exp(z)
    frule_test(DistributionsAD.fdistlogpdf, (x̃, Δx), (ỹ, Δy), (z̃, Δz))
    rrule_test(DistributionsAD.fdistlogpdf, Δu, (x̃, x̄), (ỹ, ȳ), (z̃, z̄))

    x̃ = exp(x)
    frule_test(DistributionsAD.tdistlogpdf, (x̃, Δx), (y, Δy))
    rrule_test(DistributionsAD.tdistlogpdf, Δu, (x̃, x̄), (y, ȳ))

    x̃ = rand(1:100)
    ỹ = logistic(y)
    z̃ = rand(1:x̃)
    frule_test(DistributionsAD.binomlogpdf, (x̃, nothing), (ỹ, Δy), (z̃, nothing))
    rrule_test(DistributionsAD.binomlogpdf, Δu, (x̃, nothing), (ỹ, ȳ), (z̃, nothing))

    x̃ = exp(x)
    ỹ = rand(1:100)
    frule_test(DistributionsAD.poislogpdf, (x̃, Δx), (ỹ, nothing))
    rrule_test(DistributionsAD.poislogpdf, Δu, (x̃, x̄), (ỹ, nothing))
end
