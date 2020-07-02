using .Tracker: TrackedMatrix, track, @grad, data
to_posdef(A::TrackedMatrix) = track(to_posdef, A)
Tracker.@grad function to_posdef(A::TrackedMatrix)
    data_A = data(A)
    S = data_A * data_A' + I
    function pullback(∇)
        return ((∇ + ∇') * data_A,)
    end
    return S, pullback
end
