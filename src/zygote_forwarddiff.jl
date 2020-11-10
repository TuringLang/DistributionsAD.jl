# Zygote loads ForwardDiff, so this adjoint will autmatically be loaded together
# with `using Zygote`.

# TODO: add adjoints without ForwardDiff
@adjoint function poissonbinomial_pdf_fft(x::AbstractArray{T}) where T<:Real
    fft = poissonbinomial_pdf_fft(x)
    return  fft, Δ -> begin
        ((ForwardDiff.jacobian(poissonbinomial_pdf_fft, x)::Matrix{T})' * Δ,)
    end
end

if isdefined(Distributions, :poissonbinomial_pdf)
    @adjoint function Distributions.poissonbinomial_pdf(x::AbstractArray{T}) where T<:Real
        value = Distributions.poissonbinomial_pdf(x)
        function poissonbinomial_pdf_pullback(Δ)
            return ((ForwardDiff.jacobian(Distributions.poissonbinomial_pdf, x)::Matrix{T})' * Δ,)
        end
        return value, poissonbinomial_pdf_pullback
    end
end
