function V(beta::Vector, X::Matrix)
    return X*beta
end

function V_i(beta::Vector, X::Matrix, i::Int64)
    return X[i,:]'*beta
end

function ∇V(beta::Vector, X::Matrix)
    return X
end

function ∇V_i(beta::Vector, X::Matrix, i::Int64)
    return X[i,:]
end

LU = LinearUtilitie(V, ∇V, V_i, ∇V_i)
