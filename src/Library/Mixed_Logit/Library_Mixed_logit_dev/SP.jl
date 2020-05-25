function SP(θ::Vector, ind::Individual{Array{Float64,2}}, U::Utilities, R::Int64, dim_γ::Int64, stream)
    
    reset_substream!(stream)
    value = 0.0
    γ = zeros(dim_γ)
    for i in 1:R
        γ[:] = [rand(stream) for i in 1:dim_γ]
        value += logit(θ, γ, ind, U)
    end
    value *= 1/R
    return value
end

function ∇SP(θ::Vector, ind::Individual{Array{Float64,2}}, U::Utilities, R::Int64, 
        dim_γ::Int64, stream)
    
    reset_substream!(stream)
    grad = zeros(length(θ))
    γ = zeros(dim_γ)
    for i in 1:R
        γ[:] = [rand(stream) for i in 1:dim_γ ]
        grad += ∇logit(θ, γ, ind, U)
    end
    return (1/R)*grad
end

function HSP(θ::Vector, ind::Individual{Array{Float64,2}}, U::Utilities, R::Int64, 
        dim_γ::Int64, stream)
    reset_substream!(stream)
    
    H = zeros(length(θ), length(θ))
    
    γ = zeros(dim_γ)
    for i in 1:R
        γ[:] = [rand(stream) for i in 1:dim_γ ]
        
        H += Hlogit(θ, γ, ind, U)
    end
    H*=1/R
    
    return H
end
