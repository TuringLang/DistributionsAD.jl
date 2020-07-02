module PosDef

export to_posdef, to_posdef_diagonal

using Requires
using LinearAlgebra

# Create positive definite matrix
to_posdef(A::AbstractMatrix) = A * A' + I
to_posdef_diagonal(a::AbstractVector) = Diagonal(a.^2 .+ 1)

function __init__()
    # Define adjoints for Tracker
    @require Tracker="9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c" include("posdef.jl")
    #     using .Tracker: TrackedMatrix, track, @grad, data
    #     to_posdef(A::TrackedMatrix) = track(to_posdef, A)
    #     Tracker.@grad function to_posdef(A::TrackedMatrix)
    #         data_A = data(A)
    #         S = data_A * data_A' + I
    #         function pullback(∇)
    #             return ((∇ + ∇') * data_A,)
    #         end
    #         return S, pullback
    #     end
    # end
end

end
