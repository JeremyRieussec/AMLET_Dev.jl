using Distributions

function get_cov_matrix(X::Vector, H::Matrix, J::Matrix)
    inv = H^(-1)
    return -inv*J*inv
end


function CI(X::Vector, H::Matrix, J::Matrix, α::Float64, n_ind::Int64)
    q_α = quantile(TDist(n_ind), 1-(1-α)/2)
    
    Σ = get_cov_matrix(X, H, J)
    intervalles = Array{Tuple{Float64,Float64}}(length(X))
    for i in 1:length(X)
        intervalles[i] = (X[i] - q_α*sqrt(Σ[i,i]/n_ind), X[i] + q_α*sqrt(Σ[i,i]/n_ind))
    end
    return intervalles
end
