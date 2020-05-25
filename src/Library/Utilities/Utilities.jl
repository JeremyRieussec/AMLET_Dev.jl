abstract type Utilities end

abstract type LinearUtilities <: Utilities end

struct LinearUtilitie <: LinearUtilities
    V::Function
    ∇V::Function
    V_i::Function
    ∇V_i::Function
end

