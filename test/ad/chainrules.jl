@testset "chainrules" begin
    x = randn()
    z = x + exp(randn())
    y = z + exp(randn())
    test_frule(DistributionsAD.uniformlogpdf, x, y, z)
    test_rrule(DistributionsAD.uniformlogpdf, x, y, z)
end
