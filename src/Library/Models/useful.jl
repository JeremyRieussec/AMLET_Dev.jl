using Distributions



#------    Build the confidence interval of a models m at a point X   ---------
function get_cov_matrix(m::Models, X::Vector)
    H = zeros(length(X), length(X))
    m.Hf!(X, H)

    J = zeros(length(X), length(X))
    m.bhhh!(X, J)

    inv = H^(-1)
    return inv*J*inv
end

#------------------  Intervalle de confiance    ---------------
function CI(m::Models, X::Vector, α::Float64)
    q_α = quantile(TDist(m.n_ind_total), 1-(1-α)/2)

    Σ = get_cov_matrix(m, X)

    intervalles = Array{Tuple{Float64,Float64}}(undef, length(X))
    for i in 1:length(X)

        intervalles[i] = (X[i] - q_α*sqrt(Σ[i,i]/m.n_ind_total), X[i] + q_α*sqrt(Σ[i,i]/m.n_ind_total))

    end
    return intervalles
end
