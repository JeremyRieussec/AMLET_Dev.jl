# Structure pour Logit Models (LM)

mutable struct LM <: Models  #logit model
    batch::Batch
    dim::Int64

    f::Function              
    ∇f!::Function
    Hf!::Function
    bhhh!::Function     #it's bhhh approximation of the hessian
    Lbhhh!::Function

    score!::Function

    x::Vector                #value of the optimal parameters completed by a solve methods

    undef_Field::Any         #this field is let to the user. They can put wathever the fuck they want in it.
                             #a wise user using an second order method with hessian approximation could put the
                             #final hessian approximation in it for example.
                             #this fields can be multiplicate by making it an array of Any.
                             #if you dont like it, just remake a subtype of Models that dosent contains it, wathever


    function LM(b::Batch, dim::Int64)
        lm = new()
        lm.dim = dim
        lm.batch = b
        return lm
    end
end
