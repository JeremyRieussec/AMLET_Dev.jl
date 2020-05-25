function SLL(θ::Vector, it::Batch, U::Utilities, R::Int64, dim_γ::Int64)
    val = 0.0
    total = 0
    stream = it.stream
    for ind in it
        next_substream!(stream)
        val += ind.ind.n_sim*log(SP(θ, ind.ind, U, R, dim_γ, ind.rng))
        total += ind.ind.n_sim
    end
    #println(total)
    return -val/total
end

function ∇SLL(θ::Vector, it::Batch, U::Utilities, R::Int64, dim_γ::Int64)
    
    grad = zeros(length(θ))
    total = 0
    
    for ind in it
        next_substream!(stream)
        grad_sp_i = ∇SP(θ, ind.ind, U, R, dim_γ, ind.rng)
        sp = SP(θ, ind.ind, U, R, dim_γ, ind.rng)
        grad += (ind.ind.n_sim/sp)*grad_sp_i
        total += ind.ind.n_sim
    end
    return -(1/total)*grad
end

function HSLL(θ::Vector, it::Batch, U::Utilities, R::Int64, dim_γ::Int64)
    
    total = 0
    stream = it.stream
    
    H = zeros(length(θ), length(θ))
    
    for ind in it
        next_substream!(stream)
        
        grad_sp_i = ∇SP(θ, ind.ind, U, R, dim_γ, ind.rng)
        sp = SP(θ, ind.ind, U, R, dim_γ, ind.rng)
        Hsp_i = HSP(θ, ind.ind, U, R, dim_γ, ind.rng)
        
        H += (ind.n_sim/(sp*sp))*(sp*Hsp_i - grad_sp_i*grad_sp_i')
        total += ind.ind.n_sim
        
        
    end
    H *= -1/total
    return H
end