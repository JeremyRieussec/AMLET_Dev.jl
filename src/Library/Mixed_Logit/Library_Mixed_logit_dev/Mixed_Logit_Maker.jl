function complete_Model!(mlm::MLM, U::Utilities, R::Int64, size_draw::Int64)
    batch = mlm.batch
    
    function F(β::Array{Float64, 1}, b::Batch = batch)
        return SLL(β, b, U, R, size_draw)
    end
    
    function ∇F!(β::Array{Float64, 1}, stack::Array{Float64, 1}, b::Batch = batch)
        stack[:] = ∇SLL(β, b, U, R, size_draw)
    end
    
    function HF!(β::Array{Float64, 1}, stack::Array{Float64, 2}, b::Batch = batch)
        HSLL!(β, b, stack, U, R, size_draw)
    end
    
    mlm.f = F              
    mlm.∇f! = ∇F!
    mlm.Hf! = HF!
    
end