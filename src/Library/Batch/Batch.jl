using RDST

"""
'abstract type Batch end'
"""
abstract type Batch end

"""
'abstract type BatchLM end'
# Methods:
### iterate(::BatchLM)
### iterate(::BatchLM, state)

both iterate methods have to return Union{( ind <: Individual{T}, state::Any), nothing}
"""
abstract type BatchLM <: Batch end

"""
'struct BatchMLM{S <: AbstractStreamableRNG} end'
#Fields:
### batch <: Batch
An iterable struct that return Individual{T}
### stream <: AbstractStreamableRNG
stream used in the monte carlos estimation
"""

struct BatchMLM{S <: AbstractStreamableRNG} <: Batch
    batch::Batch
    stream::S
    function BatchMLM(b::Batch, s::S) where S
        return new{S}(b,s)
    end
end

#abstract type BatchMLM{S <: AbstractStreamableRNG} end
