#logit return the value 
#∇logit! and Hlogit! add the required value to the stack, they do not replace it

function logit(θ::Vector, γ::Vector, ind::Individual{Array{Float64,2}}, U::Utilities)
    #println(γ)
    uti = U.V(θ, ind.data, γ)
    v_plus = maximum(uti)
    uti -= v_plus*ones(length(uti))
    map!(exp, uti, uti)
    s = sum(uti)
    return uti[ind.choice]/s
end

function ∇logit(θ::Vector, γ::Vector, ind::Individual{Array{Float64,2}}, U::Utilities)
    uti = U.V(θ, ind.data, γ)
    v_plus = maximum(uti)
    uti -= v_plus*ones(length(uti))
    map!(exp, uti, uti)
    s = sum(uti)
    uti *= (1/s)
    
    grad = uti[ind.choice]*U.∇V_i(θ, ind.data, γ, ind.choice)
    
    for i in 1:length(uti)
        grad -= uti[ind.choice]*uti[i]*U.∇V_i(θ, ind.data, γ, i)
    end
    
    return grad
end


function Hlogit(θ::Vector, γ::Vector, ind::Individual{Array{Float64,2}}, U::LinearUtilities)
    #println(γ)
    uti = U.V(θ, ind.data, γ)
    v_plus = maximum(uti)
    uti -= v_plus*ones(length(uti))
    map!(exp, uti, uti)
    s = sum(uti)
    uti *= (1/s) #uti contains logit probabilities
    choice = ind.choice
    
    L = zeros(length(θ))
    for i in 1:length(uti)
        L+=uti[i]*U.∇V_i(θ, ind.data, γ, choice)
    end
    
    tmp = U.∇V_i(θ, ind.data, γ, choice) - L
    Hes = uti[choice]*tmp*tmp'
    
    for i in 1:length(uti)
        Hes -= uti[choice]*uti[i]*U.∇V_i(θ, ind.data, γ, choice)*U.∇V_i(θ, ind.data, γ, choice)'
    end
    
    Hes += uti[choice]*L*L'
    return Hes
end