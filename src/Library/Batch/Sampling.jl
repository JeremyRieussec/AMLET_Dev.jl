using RDST

## -----------      Creation Subbatch -- UNIFORME   --------------------
# Tirage uniforme avec repetition

function uniformSampling(m::Models , sampleSize::Int64 , n_max::Int64)
    mrg_gen = MRG32k3aGen([set_seed for i = 1:6])
    mrg = next_stream(mrg_gen)
    perm = [Int(ceil(ntot*rand(mrg))) for i = 1:n_max]

    features = [m.batch.features[i] for i in perm]
    labels = [m.batch.labels[i] for i in perm]
    wieghts = [m.batch.weights[i] for i in perm]

    sub_batch = BatchMLP(features, labels, weights) # cf MLPBatch.jl

    b = SubBatch(sub_batch, sampleSize, n_max)

    return b
end
