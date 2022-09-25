macro register(dist)
    return quote
        DistributionsAD.eval(getexpr($(esc(dist))))
        DistributionsAD.toflatten(::$(esc(dist))) = true
    end
end
function getexpr(Tdist)
    x = gensym()
    fnames = fieldnames(Tdist)
    flattened_args = Expr(:tuple, [:(dist.$f) for f in fnames]...)
    func = Expr(:->,
                Expr(:tuple, fnames..., x),
                Expr(:block,
                    Expr(:call, :logpdf,
                        Expr(:call, :($Tdist), fnames...),
                        x,
                    )
                )
            )
    return :(flatten(dist::$Tdist) = ($func, $flattened_args))
end
const flattened_dists = [   Bernoulli,
                            BetaBinomial,
                            Binomial,
                            Geometric,
                            NegativeBinomial,
                            Poisson,
                            Skellam,
                            Arcsine,
                            BetaPrime,
                            Biweight,
                            Cauchy,
                            Chernoff,
                            Chi,
                            Chisq,
                            Cosine,
                            Epanechnikov,
                            Erlang,
                            Exponential,
                            FDist,
                            Frechet,
                            GeneralizedExtremeValue,
                            GeneralizedPareto,
                            Gumbel,
                            InverseGaussian,
                            Kolmogorov,
                            Laplace,
                            Levy,
                            Distributions.AffineDistribution,
                            Logistic,
                            LogitNormal,
                            LogNormal,
                            Normal,
                            Pareto,
                            PGeneralizedGaussian,
                            Rayleigh,
                            SymTriangularDist,
                            TDist,
                            TriangularDist,
                            Triweight,
                        ]
for T in flattened_dists
    @eval toflatten(::$T) = true
end
toflatten(::Distribution) = false
for T in flattened_dists
    eval(getexpr(T))
end
