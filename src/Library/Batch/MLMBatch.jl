

"""
'iterate(b::BatchMLM{S}, state = b.InitialState)'
### The BatchMLM{S}, will be define from the batch iterator it contains
"""

function iterate(b::BatchMLM{S}) where S
    next_ind, next_state = iterate(b.batch)
    reset_stream!(b.stream)      #we restart to the first substream
    return MLM_Individual(next_ind, b.stream), next_state
end

function iterate(b::BatchMLM{S}, state) where S
    tmp = iterate(b.batch, state)
    if tmp == nothing
        return nothing
    else
        next_ind, next_state = tmp
        
        next_Rng = next_substream!(b.stream)
        next_ind_MLM = MLM_Individual(next_ind, b.stream)
        return next_ind_MLM, next_state
    end
end