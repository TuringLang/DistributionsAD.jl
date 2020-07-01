# FIXME: This is inefficient, replace with the commented code below once Zygote supports it.
@adjoint function poissonbinomial_pdf_fft(x::AbstractArray{T}) where T<:Real
    fft = poissonbinomial_pdf_fft(x)
    return  fft, Δ -> begin
        ((ForwardDiff.jacobian(x -> poissonbinomial_pdf_fft(x), x)::Matrix{T})' * Δ,)
    end
end

# The code below doesn't work because of bugs in Zygote. The above is inefficient.
#=
ZygoteRules.@adjoint function poissonbinomial_pdf_fft(x::AbstractArray{<:Real})
    return ZygoteRules.pullback(poissonbinomial_pdf_fft_zygote, x)
end
function poissonbinomial_pdf_fft_zygote(p::AbstractArray{T}) where {T <: Real}
    n = length(p)
    ω = 2 * one(T) / (n + 1)
    lmax = ceil(Int, n/2)
    x1 = [one(T)/(n + 1)]
    x_lmaxp1 = map(1:lmax) do l
        logz = zero(T)
        argz = zero(T)
        for j=1:n
            zjl = 1 - p[j] + p[j] * cospi(ω*l) + im * p[j] * sinpi(ω * l)
            logz += log(abs(zjl))
            argz += atan(imag(zjl), real(zjl))
        end
        dl = exp(logz)
        return dl * cos(argz) / (n + 1) + dl * sin(argz) * im / (n + 1)
    end
    x_lmaxp2_end = [conj(x[l + 1]) for l in lmax:-1:1 if n + 1 - l > l]
    x = vcat(x1; x_lmaxp1, x_lmaxp2_end)
    y = [sum(x[j] * cis(-π * float(T)(2 * mod(j * k, n)) / n) for j in 1:n) for k in 1:n]
    return max.(0, real.(y))
end
function poissonbinomial_pdf_fft_zygote2(p::AbstractArray{T}) where {T <: Real}
    n = length(p)
    ω = 2 * one(T) / (n + 1)
    x = Vector{Complex{T}}(undef, n+1)
    lmax = ceil(Int, n/2)
    x[1] = one(T)/(n + 1)
    for l=1:lmax
        logz = zero(T)
        argz = zero(T)
        for j=1:n
            zjl = 1 - p[j] + p[j] * cospi(ω*l) + im * p[j] * sinpi(ω * l)
            logz += log(abs(zjl))
            argz += atan(imag(zjl), real(zjl))
        end
        dl = exp(logz)
        x[l + 1] = dl * cos(argz) / (n + 1) + dl * sin(argz) * im / (n + 1)
        if n + 1 - l > l
            x[n + 1 - l + 1] = conj(x[l + 1])
        end
    end
    max.(0, real.(_dft_zygote(copy(x))))
end
function _dft_zygote(x::Vector{T}) where T
    n = length(x)
    y = Zygote.Buffer(zeros(complex(float(T)), n))
    @inbounds for j = 0:n-1, k = 0:n-1
        y[k+1] += x[j+1] * cis(-π * float(T)(2 * mod(j * k, n)) / n)
    end
    return copy(y)
end
=#
