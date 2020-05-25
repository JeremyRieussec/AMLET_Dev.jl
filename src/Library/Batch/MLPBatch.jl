
#----------     Structure Batch Multi Layer Perceptron      -----------------
mutable struct BatchMLP <: Batch
    features::Matrix # -- X : matrice des vecteurs entree = (x_1 | x_2 | ... | x_n)
    labels::Matrix # -- Y : matrice des vecteurs label sortie = (y_1 | y_2 | ... | y_n)
    train_x::SubArray # training Xset
    train_y::SubArray # training Yset
    test_x::SubArray # test Xset
    test_y::SubArray # test Yset
    valid_x::SubArray # validation Xset
    valid_y::SubArray # validation Yset
    weights::Array # poids de chaque individu dans le batch

    # constructeur BatchMLP
    function BatchMLP(features, labels, weights)
        b = new()
        b.features = features
        b.labels = labels
        b.weights = weights
        return b
    end
end


#-------------      iterate         -----------------------
function iterate(b::BatchMLP, state::Int64 = 1)
    if state > size(b.train_y, 2) # taille echantillon
        return nothing
    else
        # retour :
        #   - LM_indvidu(individu[state] , son choix/label , nombre similaire)
        #   - state + 1
        return LM_Individual(b.train_x[:, state], argmax(b.train_y[:, state])-1, b.weights[state]), state+1
    end
end
