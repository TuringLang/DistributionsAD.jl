# Zygote loads ForwardDiff, so this adjoint will autmatically be loaded together
# with `using Zygote`.
@adjoint function Distributions.poissonbinomial_pdf(x::AbstractArray{T}) where T<:Real
    value = Distributions.poissonbinomial_pdf(x)
    function poissonbinomial_pdf_pullback(Δ)
        return ((ForwardDiff.jacobian(Distributions.poissonbinomial_pdf, x)::Matrix{T})' * Δ,)
    end
    return value, poissonbinomial_pdf_pullback
end
