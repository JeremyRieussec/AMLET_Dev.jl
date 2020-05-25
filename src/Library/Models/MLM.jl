
#------------- Structure pour Mixed Logit Models (MLM)  --------------

mutable struct MLM <: Models
    batch::Batch
    dim::Int64
    f::Function
    ∇f!::Function
    Hf!::Function
    bhhh!::Function     #it's bhhh approximation of the hessian
    Lbhhh!::Function    #take β and return all the gradient as row

    Bias::Function
    ∇Bias!::Function
    Hbias!::Function

    x::Vector                #value of the optimal parameters completed by a solve methods
    undefine                 #wathever you want


    function MLM(batch::Batch, dim::Int64)
        mlm = new()
        mlm.dim = dim
        mlm.batch = batch
        return mlm
    end
end
