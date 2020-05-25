function V(θ::Vector, X_i::Matrix, γ::Vector)
    β = θ[1:Int64(length(θ)/2)] + Diagonal(γ)*θ[Int64(length(θ)/2)+1:end]
    return X_i*β
end

function ∇V(θ::Vector, X_i::Matrix, γ::Vector)
    return [X_i X_i*Diagonal(γ)]
end

function V_i(θ::Vector, X_i::Matrix, γ::Vector, i::Int64)
    β = θ[1:Int64(length(θ)/2)] + Diagonal(γ)*θ[Int64(length(θ)/2)+1:end]
    return X_i[i,:]'*β
end

function ∇V_i(θ::Vector, X_i::Matrix, γ::Vector, i::Int64)
    return ([X_i X_i*Diagonal(γ)])[i,:]
end

UVINLU = LinearUtilitie(V, ∇V, V_i, ∇V_i)