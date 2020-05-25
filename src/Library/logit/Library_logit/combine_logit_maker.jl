function log_logit(β::Vector, ind::Individual{T}, U::Utilities, weight::Int64) where T
    uti = U.V(β, ind.data)
    v_plus = maximum(uti)
    uti -= v_plus*ones(length(uti))

    map!(exp, uti, uti)
    s = sum(uti)
    return -weight*log(uti[ind.choice]/s)
end

function ∇log_logit!(β::Vector, ind::Individual{T}, stack::Vector, U::Utilities, weight::Int64) where T
    
    uti = U.V(β, ind.data)
    v_plus = maximum(uti)
    uti -= v_plus*ones(length(uti))

    map!(exp, uti, uti)
    s = sum(uti)
    
    uti *= 1/s 
    #println(ind.data)
    stack[:] -= weight*U.∇V_i(β, ind.data, ind.choice)
    
    stack[:] += weight*sum(uti[i]*U.∇V_i(β, ind.data, i) for i in 1:length(uti))
    
end

function Hlog_logit!(β::Vector, ind::Individual{T}, stack::Matrix, U::Utilities, weight::Int64)  where T
    uti = U.V(β, ind.data)
    v_plus = maximum(uti)
    uti -= v_plus*ones(length(uti))

    map!(exp, uti, uti)
    s = sum(uti)
    
    uti *= 1/s 
    
    sum_F = zeros(length(β))
    
    for j in 1:length(uti)
        stack[:,:] += weight*uti[j]*U.∇V_i(β, ind.data, j)*U.∇V_i(β, ind.data, j)'
        
        stack[:,:] += weight*uti[j]*U.HV_i(β, ind.data, j)
        
        sum_F += uti[j]*U.∇V_i(β, ind.data, j)
    end
    
    stack[:,:] -= weight*sum_F*sum_F'
    stack[:,:] -= weight*U.HV_i(β, ind.data, ind.choice)
end  
    
    
#Linear Utilities catched here
function Hlog_logit!(β::Vector, ind::Individual{T}, stack::Matrix, U::LinearUtilities, weight::Int64) where T
    uti = U.V(β, ind.data)
    v_plus = maximum(uti)
    uti -= v_plus*ones(length(uti))

    map!(exp, uti, uti)
    s = sum(uti)
    
    uti *= 1/s 
    
    sum_F = zeros(length(β))
    
    for j in 1:length(uti)
        stack[:,:] += weight*uti[j]*U.∇V_i(β, ind.data, j)*U.∇V_i(β, ind.data, j)'
        sum_F += uti[j]*U.∇V_i(β, ind.data, j)
    end
    
    stack[:,:] -= weight*sum_F*sum_F'
end

function bhhh!(β::Vector, ind::Individual{T}, stack::Matrix, U::LinearUtilities, weight::Int64) where T
    uti = U.V(β, ind.data)
    v_plus = maximum(uti)
    uti -= v_plus*ones(length(uti))

    map!(exp, uti, uti)
    s = sum(uti)
    
    uti *= 1/s 
    grad = -ind.data[ind.choice, :] + sum(uti[i]*U.∇V_i(β, ind.data, i) for i in 1:length(uti))
    
    stack[:,:] += weight*grad*grad'
end

function ALL(β::Vector, it::Batch, U::Utilities)
    value = 0.0
    total = 0
    for ind in it
        total += ind.n_sim
        value += log_logit(β, ind, U, ind.n_sim)
    end
    return value/total
end

function ∇ALL!(β::Vector, it::Batch, stack::Vector, U::Utilities)
    stack[:] = zeros(length(β))
    total = 0
    for ind in it
        total += ind.n_sim
        ∇log_logit!(β, ind, stack, U, ind.n_sim)
    end
    stack[:,:] *= 1/total
end

function HALL!(β::Vector, it::Batch, stack::Matrix, U::Utilities)#hessian of average log likelihood
    stack[:, :] = zeros(length(β), length(β))
    total = 0
    for ind in it
        total += ind.n_sim
        Hlog_logit!(β, ind, stack, U, ind.n_sim)
    end
    stack[:,:] *= 1/total
end

function BHHHALL!(β::Vector, it::Batch, stack::Matrix, U::Utilities)
    stack[:, :] = zeros(length(β), length(β))
    total = 0
    for ind in it
        total += ind.n_sim
        bhhh!(β, ind, stack, U, ind.n_sim)
    end
    stack[:,:] *= 1/total
end

function computeScores!(β::Vector, it::Batch, stack::Vector{Float64}, score::Array{Vector{Float64}}, U::Utilities)
    #j = 1
    stack[:] = zeros(length(stack))
    total = 0
    for (k, ind) in enumerate(it)
        total += ind.n_sim
        
        uti = U.V(β, ind.data)
        v_plus = maximum(uti)
        uti -= v_plus*ones(length(uti))

        map!(exp, uti, uti)
        s = sum(uti)
    
        uti *= 1/s 
        grad = -ind.data[ind.choice, :] + sum(uti[i]*U.∇V_i(β, ind.data, i) for i in 1:length(uti))
        score[k] = copy(grad)
        stack[:] += ind.n_sim*grad
        #stack[j:j+ind.n_sim-1] = [grad for l in 1:ind.n_sim]
        #j += ind.n_sim
    end
    stack[:] *= 1/total
end



function complete_Model!(lm::LM, U::Utilities)
    batch = lm.batch
    
    function F(β::Array{Float64, 1}, b::Batch = batch)
        return ALL(β, b, U)
    end
    
    function ∇F!(β::Array{Float64, 1}, stack::Array{Float64, 1}, b::Batch = batch)
        ∇ALL!(β, b, stack, U)
    end
    
    function HF!(β::Array{Float64, 1}, stack::Array{Float64, 2}, b::Batch = batch)
        HALL!(β, b, stack, U)
    end
    
    function bhhh!(β::Array{Float64, 1}, stack::Array{Float64, 2}, b::Batch = batch)
        BHHHALL!(β, b, stack, U)
    end
    
    function score!(β::Array{Float64, 1}, stack::Vector{Float64}, score::Array{Vector{Float64}}, b::Batch = batch)
        computeScores!(β, b, stack, score, U)
    end
    
    lm.f = F              
    lm.∇f! = ∇F!
    lm.Hf! = HF!
    
    lm.bhhh! = bhhh!
    
    lm.score! = score!
end
