
#-------------      Structure SubBatch      -------------------------------
mutable struct SubBatch <: Batch
    batch::Batch # ensemble des donnees du SubBatch
    size::Int64 # taille echantillon
    n_ind::Int64 #
    state_next # pour iterate

    # Constructeur SubBatch
    function SubBatch(batch, n::Int64, n_max::Int64)
        b = new()
        b.batch = batch
        b.size = n
        return b
    end
end

#--------------     iterate SubBatch        ---------------------------
function iterate(b::SubBatch, state::Int64 = 1)
    if state > b.size
        return nothing
    else
        tmp = iterate(b.batch, b.state_next)
        if tmp == nothing
            ind, st = iterate(b.batch) # recommence iterate au depart
            b.state_next = st
            return ind, state +1
        else
            ind, st = tmp
            b.state_next = st
            return ind, state +1
        end
    end
end
