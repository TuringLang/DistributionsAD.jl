## Uniform ##

@scalar_rule(
    uniformlogpdf(a, b, x),
    @setup(
        insupport = a <= x <= b,
        diff = b - a,
        c = insupport ? inv(diff) : inv(one(diff)),
    ),
    (c, -c, ZeroTangent()),
)

# The following rules should be moved to Distributions
# To ensure that this transfer does not break DistributionsAD we check if Distributions uses CR
if !isdefined(Distributions, :ChainRulesCore)

## PoissonBinomial
function ChainRulesCore.rrule(
    ::typeof(Distributions.poissonbinomial_pdf_fft), p::AbstractVector{<:Real}
)
    y = Distributions.poissonbinomial_pdf_fft(p)
    A = poissonbinomial_partialderivatives(p)
    function poissonbinomial_pdf_fft_pullback(Δy)
        p̄ = InplaceableThunk(
            Δ -> LinearAlgebra.mul!(Δ, A, Δy, true, true),
            @thunk(A * Δy),
        )
        return (NoTangent(), p̄)
    end
    return y, poissonbinomial_pdf_fft_pullback
end

if isdefined(Distributions, :poissonbinomial_pdf)
    function ChainRulesCore.rrule(
        ::typeof(Distributions.poissonbinomial_pdf), p::AbstractVector{<:Real}
    )
        y = Distributions.poissonbinomial_pdf(p)
        A = poissonbinomial_partialderivatives(p)
        function poissonbinomial_pdf_pullback(Δy)
            p̄ = InplaceableThunk(
                Δ -> LinearAlgebra.mul!(Δ, A, Δy, true, true),
                @thunk(A * Δy),
            )
            return (NoTangent(), p̄)
        end
        return y, poissonbinomial_pdf_pullback
    end
end

end
