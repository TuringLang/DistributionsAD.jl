# A lot of this module is adapted from Tracker.jl and ReverseDiff.jl
# ReverseDiff.jl is not actively developed but it would be nice to move the code in this 
# module to ReverseDiff at some point.

###############
## logsumexp ##
###############

logsumexp(x::TrackedArray; dims=:) = track(logsumexp, x, dims = dims)
@grad function logsumexp(x::AbstractArray; dims)
    x_value = value(x)
    lse = logsumexp(x_value; dims=dims)
    return lse, Δ -> (Δ .* exp.(x_value .- lse),)
end

############
## linalg ##
############

function LinearAlgebra.cholesky(A::Symmetric{<:Any, <:TrackedMatrix}; check=true)
    uplo = A.uplo == 'U' ? (:U) : (:L)
    factors, info = symm_turing_chol(parent(A), check, uplo)
    return Cholesky{eltype(factors), typeof(factors)}(factors, 'U', info)
end
function LinearAlgebra.cholesky(A::TrackedMatrix; check=true)
    factors, info = turing_chol(A, check)
    return Cholesky{eltype(factors), typeof(factors)}(factors, 'U', info)
end

function symm_turing_chol(x::TrackedArray{V,D}, check, uplo) where {V,D}
    tp = tape(x)
    x_value = value(x)
    (factors,info), back = DistributionsAD.symm_turing_chol_back(x_value, check, uplo)
    C = Cholesky{eltype(factors), typeof(factors)}(factors, 'U', info)
    out = track(C.factors, D, tp)
    record!(tp, SpecialInstruction, symm_turing_chol, (x, check, uplo), out, (back, issuccess(C)))
    return out, C.info
end
function turing_chol(x::TrackedArray{V,D}, check) where {V,D}
    tp = tape(x)
    x_value = value(x)
    (factors,info), back = DistributionsAD.turing_chol_back(x_value, check)
    C = Cholesky{eltype(factors), typeof(factors)}(factors, 'U', info)
    out = track(C.factors, D, tp)
    record!(tp, SpecialInstruction, turing_chol, (x, check), out, (back, issuccess(C)))
    return out, C.info
end

for f in (:turing_chol, :symm_turing_chol)
    @eval begin
        @noinline function ReverseDiff.special_reverse_exec!(
            instruction::SpecialInstruction{typeof($f)},
        )
            output = instruction.output
            instruction.cache[2] || throw(PosDefException(C.info))
            input = instruction.input
            input_deriv = deriv(input[1])
            P = instruction.cache[1]
            input_deriv .+= P((factors = deriv(output),))[1]
            unseed!(output)
            return nothing
        end
    end
end

@noinline function ReverseDiff.special_forward_exec!(
    instruction::SpecialInstruction{typeof(turing_chol)},
)
    output, input = instruction.output, instruction.input
    factors = turing_chol(value.(input)...)[1]
    value!(output, factors)
    return nothing
end

@noinline function ReverseDiff.special_forward_exec!(
    instruction::SpecialInstruction{typeof(symm_turing_chol)},
)
    output, input = instruction.output, instruction.input
    factors = symm_turing_chol(value.(input)...)[1]
    value!(output, factors)
    return nothing
end
