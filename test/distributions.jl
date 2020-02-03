dim = 3
mean = zeros(dim)
cov_mat = Matrix{Float64}(I, dim, dim)
cov_vec = ones(dim)
cov_num = 1.0
norm_val_vec = ones(dim)
norm_val_mat = ones(dim, 2)
alpha = ones(4)
dir_val = fill(0.25, 4)
beta_mat = rand(MatrixBeta(dim, dim, dim))
tested = []
function test_info(name)
    if !(name in tested)
        push!(tested, name)
        @info("Testing: $(name)")
    end
end
test_head(s) = println("\n"*s*"\n")
separator() = println("\n"*"="^50)

separator()
@testset "Univariate discrete distributions" begin
    test_head("Testing: Univariate discrete distributions")
    uni_disc_dists = [
        DistSpec(:Bernoulli, (0.45,), 1),
        DistSpec(:Bernoulli, (0.45,), 0),
        DistSpec(:((a, b) -> BetaBinomial(10, a, b)), (2, 1), 5),
        DistSpec(:(p -> Binomial(10, p)), (0.5,), 5),
        DistSpec(:Categorical, ([0.45, 0.55],), 1),
        DistSpec(:Geometric, (0.45,), 3),
        DistSpec(:NegativeBinomial, (3.5, 0.5), 1),
        DistSpec(:Poisson, (0.5,), 1),
        DistSpec(:Skellam, (1.0, 2.0), -2),
        DistSpec(:PoissonBinomial, ([0.5, 0.5],), 0),
    ]
    for d in uni_disc_dists
        test_info(d.name)
        for testf in get_all_functions(d, false)
            test_ad(testf.f, testf.x)
        end
    end
end
separator()

@testset "Univariate continuous distributions" begin
    test_head("Testing: Univariate continuous distributions")
    uni_cont_dists = [
        DistSpec(:Arcsine, (), 0.5),
        DistSpec(:Arcsine, (1,), 0.5),
        DistSpec(:Arcsine, (0, 2), 0.5),
        DistSpec(:Beta, (), 0.5),
        DistSpec(:Beta, (1,), 0.5),
        DistSpec(:Beta, (1, 2), 0.5),
        DistSpec(:BetaPrime, (), 0.5),
        DistSpec(:BetaPrime, (1,), 0.5),
        DistSpec(:BetaPrime, (1, 2), 0.5),
        DistSpec(:Biweight, (), 0.5),
        DistSpec(:Biweight, (1,), 0.5),
        DistSpec(:Biweight, (1, 2), 0.5),
        DistSpec(:Cauchy, (), 0.5),
        DistSpec(:Cauchy, (1,), 0.5),
        DistSpec(:Cauchy, (1, 2), 0.5),
        DistSpec(:Chi, (1,), 0.5),
        DistSpec(:Chisq, (1,), 0.5),
        DistSpec(:Cosine, (1, 1), 0.5),
        DistSpec(:Epanechnikov, (1, 1), 0.5),
        DistSpec(:((s)->Erlang(1, s)), (1,), 0.5), # First arg is integer
        DistSpec(:Exponential, (1,), 0.5),
        DistSpec(:FDist, (1, 1), 0.5),
        DistSpec(:Frechet, (), 0.5),
        DistSpec(:Frechet, (1,), 0.5),
        DistSpec(:Frechet, (1, 2), 0.5),
        DistSpec(:Gamma, (), 0.5),
        DistSpec(:Gamma, (1,), 0.5),
        DistSpec(:Gamma, (1, 2), 0.5),
        DistSpec(:GeneralizedExtremeValue, (1.0, 1.0, 1.0), 0.5),
        DistSpec(:GeneralizedPareto, (), 0.5),
        DistSpec(:GeneralizedPareto, (1.0, 2.0), 0.5),
        DistSpec(:GeneralizedPareto, (0.0, 2.0, 3.0), 0.5),
        DistSpec(:Gumbel, (), 0.5),
        DistSpec(:Gumbel, (1,), 0.5),
        DistSpec(:Gumbel, (1, 2), 0.5),
        DistSpec(:InverseGamma, (), 0.5),
        DistSpec(:InverseGamma, (1.0,), 0.5),
        DistSpec(:InverseGamma, (1.0, 2.0), 0.5),
        DistSpec(:InverseGaussian, (), 0.5),
        DistSpec(:InverseGaussian, (1,), 0.5),
        DistSpec(:InverseGaussian, (1, 2), 0.5),
        DistSpec(:Kolmogorov, (), 0.5),
        DistSpec(:Laplace, (), 0.5),
        DistSpec(:Laplace, (1,), 0.5),
        DistSpec(:Laplace, (1, 2), 0.5),
        DistSpec(:Levy, (), 0.5),
        DistSpec(:Levy, (0.0,), 0.5),
        DistSpec(:Levy, (0.0, 2.0), 0.5),
        DistSpec(:((a, b) -> LocationScale(a, b, Normal())), (1.0, 2.0), 0.5),
        DistSpec(:Logistic, (), 0.5),
        DistSpec(:Logistic, (1,), 0.5),
        DistSpec(:Logistic, (1, 2), 0.5),
        DistSpec(:LogitNormal, (), 0.5),
        DistSpec(:LogitNormal, (1,), 0.5),
        DistSpec(:LogitNormal, (1, 2), 0.5),
        DistSpec(:LogNormal, (), 0.5),
        DistSpec(:LogNormal, (1,), 0.5),
        DistSpec(:LogNormal, (1, 2), 0.5),
        DistSpec(:Normal, (), 0.5),
        DistSpec(:Normal, (1.0,), 0.5),
        DistSpec(:Normal, (1.0, 2.0), 0.5),
        DistSpec(:NormalCanon, (1.0, 2.0), 0.5),
        DistSpec(:NormalInverseGaussian, (1.0, 2.0, 1.0, 1.0), 0.5),
        DistSpec(:Pareto, (), 1.5),
        DistSpec(:Pareto, (1,), 1.5),
        DistSpec(:Pareto, (1, 1), 1.5),
        DistSpec(:PGeneralizedGaussian, (), 0.5),
        DistSpec(:PGeneralizedGaussian, (1, 1, 1), 0.5),
        DistSpec(:Rayleigh, (), 0.5),
        DistSpec(:Rayleigh, (1,), 0.5),
        DistSpec(:Semicircle, (1.0,), 0.5),
        DistSpec(:SymTriangularDist, (), 0.5),
        DistSpec(:SymTriangularDist, (1,), 0.5),
        DistSpec(:SymTriangularDist, (1, 2), 0.5),
        DistSpec(:TDist, (1,), 0.5),
        DistSpec(:TriangularDist, (1, 2), 1.5),
        DistSpec(:TriangularDist, (1, 3, 2), 1.5),
        DistSpec(:Triweight, (1, 1), 1),
        DistSpec(:((mu, sigma, l, u) -> truncated(Normal(mu, sigma), l, u)), (0.0, 1.0, 1.0, 2.0), 1.5),
        DistSpec(:Uniform, (), 0.5),
        DistSpec(:Uniform, (0, 1), 0.5),
        DistSpec(:VonMises, (), 1),
        DistSpec(:Weibull, (), 1),
        DistSpec(:Weibull, (1,), 1),
        DistSpec(:Weibull, (1, 1), 1),
    ]
    broken_uni_cont_dists = [
        # Zygote
        DistSpec(:Chernoff, (), 0.5),
        # Broken in Distributions even without autodiff
        DistSpec(:(()->KSDist(1)), (), 0.5), 
        DistSpec(:(()->KSOneSided(1)), (), 0.5), 
        DistSpec(:StudentizedRange, (1.0, 2.0), 0.5),
        # Dispatch error caused by ccall
        DistSpec(:NoncentralBeta, (1.0, 2.0, 1.0), 0.5), 
        DistSpec(:NoncentralChisq, (1.0, 2.0), 0.5),
        DistSpec(:NoncentralF, (1, 2, 1), 0.5),
        DistSpec(:NoncentralT, (1, 2), 0.5),
        # Stackoverflow caused by SpecialFunctions.besselix
        DistSpec(:VonMises, (1.0,), 1.0),
        DistSpec(:VonMises, (1, 1), 1),
    ]
    for d in uni_cont_dists
        test_info(d.name)
        for testf in get_all_functions(d, true)
            test_ad(testf.f, testf.x)
        end
    end
end
separator()

@testset "Multivariate discrete distributions" begin
    test_head("Testing: Multivariate discrete distributions")
    mult_disc_dists = [
        DistSpec(:((p) -> Multinomial(2, p / sum(p))), (fill(0.5, 2),), [2, 0]),
    ]
    for d in mult_disc_dists
        test_info(d.name)
        for testf in get_all_functions(d, false)
            test_ad(testf.f, testf.x)
        end
    end
end
separator()

@testset "Multivariate continuous distributions" begin
    test_head("Testing: Multivariate continuous distributions")
    mult_cont_dists = [
        # Vector case
        DistSpec(:MvNormal, (mean, cov_mat), norm_val_vec),
        DistSpec(:MvNormal, (mean, cov_vec), norm_val_vec),
        DistSpec(:MvNormal, (mean, Diagonal(cov_vec)), norm_val_vec),
        DistSpec(:MvNormal, (mean, cov_num), norm_val_vec),
        DistSpec(:((m, v) -> MvNormal(m, v*I)), (mean, cov_num), norm_val_vec),
        DistSpec(:MvNormal, (cov_mat,), norm_val_vec),
        DistSpec(:MvNormal, (cov_vec,), norm_val_vec),
        DistSpec(:MvNormal, (Diagonal(cov_vec),), norm_val_vec),
        DistSpec(:(cov_num -> MvNormal(dim, cov_num)), (cov_num,), norm_val_vec),
        DistSpec(:MvLogNormal, (mean, cov_mat), norm_val_vec),
        DistSpec(:MvLogNormal, (mean, cov_vec), norm_val_vec),
        DistSpec(:MvLogNormal, (mean, Diagonal(cov_vec)), norm_val_vec),
        DistSpec(:MvLogNormal, (mean, cov_num), norm_val_vec),
        DistSpec(:MvLogNormal, (cov_mat,), norm_val_vec),
        DistSpec(:MvLogNormal, (cov_vec,), norm_val_vec),
        DistSpec(:MvLogNormal, (Diagonal(cov_vec),), norm_val_vec),
        DistSpec(:(cov_num -> MvLogNormal(dim, cov_num)), (cov_num,), norm_val_vec),
        # Matrix case
        DistSpec(:MvNormal, (mean, cov_vec), norm_val_mat),
        DistSpec(:MvNormal, (mean, Diagonal(cov_vec)), norm_val_mat),
        DistSpec(:MvNormal, (mean, cov_num), norm_val_mat),
        DistSpec(:((m, v) -> MvNormal(m, v*I)), (mean, cov_num), norm_val_mat),
        DistSpec(:MvNormal, (cov_vec,), norm_val_mat),
        DistSpec(:MvNormal, (Diagonal(cov_vec),), norm_val_mat),
        DistSpec(:(cov_num -> MvNormal(dim, cov_num)), (cov_num,), norm_val_mat),
        DistSpec(:MvLogNormal, (mean, cov_vec), norm_val_mat),
        DistSpec(:MvLogNormal, (mean, Diagonal(cov_vec)), norm_val_mat),
        DistSpec(:MvLogNormal, (mean, cov_num), norm_val_mat),
        DistSpec(:MvLogNormal, (cov_vec,), norm_val_mat),
        DistSpec(:MvLogNormal, (Diagonal(cov_vec),), norm_val_mat),
        DistSpec(:(cov_num -> MvLogNormal(dim, cov_num)), (cov_num,), norm_val_mat),
        DistSpec(:Dirichlet, (alpha,), dir_val),
        DistSpec(:Dirichlet, (alpha,), dir_val),
    ]

    broken_mult_cont_dists = [
        # Dispatch error
        DistSpec(:MvNormalCanon, (mean, cov_mat), norm_val_vec),
        DistSpec(:MvNormalCanon, (mean, cov_vec), norm_val_vec),
        DistSpec(:MvNormalCanon, (mean, cov_num), norm_val_vec),
        DistSpec(:MvNormalCanon, (cov_mat,), norm_val_vec),
        DistSpec(:MvNormalCanon, (cov_vec,), norm_val_vec),
        DistSpec(:(cov_num -> MvNormalCanon(dim, cov_num)), (cov_num,), norm_val_vec),
        DistSpec(:MvNormalCanon, (mean, cov_mat), norm_val_mat),
        DistSpec(:MvNormalCanon, (mean, cov_vec), norm_val_mat),
        DistSpec(:MvNormalCanon, (mean, cov_num), norm_val_mat),
        DistSpec(:MvNormalCanon, (cov_mat,), norm_val_mat),
        DistSpec(:MvNormalCanon, (cov_vec,), norm_val_mat),
        DistSpec(:(cov_num -> MvNormalCanon(dim, cov_num)), (cov_num,), norm_val_mat),
        # Test failure
        DistSpec(:MvNormal, (mean, cov_mat), norm_val_mat),
        DistSpec(:MvNormal, (cov_mat,), norm_val_mat),
        DistSpec(:MvLogNormal, (mean, cov_mat), norm_val_mat),
        DistSpec(:MvLogNormal, (cov_mat,), norm_val_mat),
        DistSpec(:(() -> Product(Normal.(randn(dim), 1))), (), norm_val_vec),
        DistSpec(:(() -> Product(Normal.(randn(dim), 1))), (), norm_val_mat),
    ]

    for d in mult_cont_dists
        test_info(d.name)
        for testf in get_all_functions(d, true)
            test_ad(testf.f, testf.x)
        end
    end
end
separator()

@testset "Matrix-variate continuous distributions" begin
    test_head("Testing: Matrix-variate continuous distributions")
    matrix_cont_dists = [
        DistSpec(:((n1, n2)->MatrixBeta(dim, n1, n2)), (dim, dim), beta_mat),
        DistSpec(:Wishart, (dim, cov_mat), cov_mat),
        DistSpec(:InverseWishart, (dim, cov_mat), cov_mat),
    ]
    broken_matrix_cont_dists = [
        # Other
        DistSpec(:MatrixNormal, (cov_mat, cov_mat, cov_mat), cov_mat),
        DistSpec(:(()->MatrixNormal(dim, dim)), (), cov_mat),
        DistSpec(:MatrixTDist, (1.0, cov_mat, cov_mat, cov_mat), cov_mat),
        DistSpec(:MatrixFDist, (dim, dim, cov_mat), cov_mat),
    ]

    for d in matrix_cont_dists
        test_info(d.name)
        for testf in get_all_functions(d, true)
            test_ad(testf.f, testf.x)
        end
    end
end
separator()
