"""
Multilayer Perceptron
Cost function : Cross-Entropy + L2 regularization
Activation function : ReLU (or Sigmoïd)
Output function : Softmax
Labels structure : One-hot encoding
Weights initilization : Normal(0,1/sqrt(sizes[l]))
"""
#--------------         Structure Network -------------------------
# -- premiere couche : vecteurs entree
# -- derniere couche : probabilites de chaque label avec SoftMax
mutable struct Network
    num_layers::Int64 # -- nombre couche reseau : au minimum 2 , input / output
    sizes::Vector{Int64} # -- taille de toutes les couches : input / output + hidden
    weights::Array{Matrix{Float64}} # -- tableau de matrices : 1 matrice represente
                                    #    les poids w_ij pour passer d'une couche a l'autre
                                    # il y aura num_layers-1 matrices de poids
    index::Array # -- indices necessaire pour representer passage couche a l'autre
    a::Array{Matrix{Float64}} # -- valeur d'Activation de chaque neurone
                              # a est un tableau de matrices :
                              #     - a[l] = matrice des valeurs des neurones pour couche l
                              # pour chaque matrice M :
                              #     - colonne M[:,j] --> valeur des neurones avec data j
                              # a[m][:,j] = valeurs des neurones couche m avec entree x_j
                              # Derniere colonne = (1 1 ... 1)^T pour prendre en compte
                              # addition du biais
    δ::Matrix{Float64} # -- pour backpropagation : represente ∇f_i(β) w.r.t. a(L)
                                                    # où L couche en cours
    λ::Float64 # -- Pour regularization

    # ------------ Constructeur Network ---------------
    # -- sizes : tableau dimension chaque couche --> input / hidde / Output
    # -- λ : facteur regularisation
    # -- n_total : taille population
    # -- seed
    function Network(sizes::Vector{Int64}, λ::Float64, n_total::Int64, seed::Int64)
        N = new()
        N.num_layers = length(sizes)
        N.sizes = copy(sizes)
        if seed > 0
            mrg_gen = MRG32k3aGen([seed for i = 1:6])
            mrg = next_stream(mrg_gen)
            N.weights = Array{Matrix{Float64}}(undef, N.num_layers-1)
        else
            # zip cree l'ensemble des couples (h , h+1) pour passage d'une couche a l'autre
            # (j , i+1) : le +1 est pour le biais
            N.weights = [zeros(j, i+1) for (i,j) in zip(sizes[1:N.num_layers-1], sizes[2:N.num_layers])]
        end
        N.index = Array{UnitRange{Int64}}(undef, N.num_layers-1)

        for k = 1:N.num_layers-1 # -- Initialisation des poids
            if seed > 0
                next_substream!(mrg)
                N.weights[k] = quantile.(Normal(0.0, 1/sqrt(sizes[k])), [rand(mrg) for i=1:sizes[k+1], j=1:sizes[k]+1])
                # Initialisation par normale N(0 , 1/size(couche))
            end
            i = sizes[k]
            j = sizes[k+1]
            # Pour passer de la couche L-1 a L, il y a besoin de (size(L-1)+1)*size(L) indices
            # On index[k] contient tranches indices pour passer de k a k+1
            N.index[k] = (k==1 ? 0 : N.index[k-1][end])+1:(k==1 ? 0 : N.index[k-1][end])+j*(i+1)
        end # --- End for poids ---

        # --- on initialise toutes les activites de neurone à 1 ----
        # enumerate(sizes) =  { (1, sizes[1]) ; (2,sizes[2]) ; ... }
        # +1 pour les couches intermediaires : pour ajouter le biais
        # +0 pour derniere couche : pas de biais
        N.a = [ones(j + (i==length(sizes) ? 0 : 1), n_total) for (i,j) in enumerate(sizes)]

        N.δ = zeros(maximum(sizes), n_total)
        N.λ = λ
        return N
    end # fin Constructeur
end

#------------------         cross_entropy!      ------------------------------

# -- N : Multi Layer Perceptron
# -- Y : Matrice des resultats de sortie attendus (y_1 | y_2 | ... | y_n)
#       - # lignes = # sorties
#       - # colonne = # data --> taille echantillon sur population
#       - colonne j : resultats de sortie attendue pour data j
#       - ligne i : resultat sortie i
# -- data : echantillon data
function cross_entropy!(N::Network, Y::AbstractArray, data::AbstractArray)
    res = Series(Mean(), Variance()) # cf package OnlineStats : serie de donnes que l'on peut mettre a jour
    # res = serie de donnes des residus pour en calculer moyenne et variance
    # ---> cross-entropy
    cov = Mean() # cf package OnlineStats : calcul de Covariance
    n = size(Y, 2) # nombre individus dans echantillon
    for i = 1:n # parcours sur les data[i]
        y = @view Y[:, i] # Cree une coupe de la matrice (colonne i)
                          #  --> y = resultat sortie attendue avec entree data i
        j = argmax(y) # donne index element le plus grand
        error = -log(N.a[end][j, i]) # a[end] : valeurs des neurones en sortie
                                     # a[end][j,i] : valeur neurone a_j sous entree data i
                                     # --> -log( P(y_j | x_i , beta) ) avec SoftMax
        fit!(res, error) # cf package OnlineStats -- Met a jour moyenne des residus
        fit!(cov, error*data[i]) # cf package OnlineStats
        data[i] = error
    end
    avg, var = value(res) # avg  = cross-entropy
                          # var = variance du vecteur d'erreurs

    # terme de regularisation L2
    L2 = @views N.λ/(2*n)*sum(sum(N.weights[i][:, 1:end-1].^2) for i = 1:N.num_layers-1)

    # resultat :
    #   - avg + L2 = valeur cross-entropy avec distribution empirique + regularization L2
    #   - var = variance des residus
    #   - value(cov) =
    return avg + L2, var, value(cov)
end

#-----------------------    cross_entropy!  ---------------------------

# ----------> Calcul cross-entropy avec comparaison aux labels de sortie Y
function cross_entropy!(N::Network, Y::AbstractArray)
    #res = Series(Mean(), Variance())
    res = Mean()
    n = size(Y, 2) # taille echantillon
    for i = 1:n
        y = @view Y[:, i] # labels de sortie pour x_i
        j = argmax(y) # j est le label de x_i
        error = -log(N.a[end][j, i]) # --> -log( P(y_j | x_i , beta) ) avec SoftMax
        fit!(res, error) # mise a jour cross entropy avec exemplaire x_i
    end
    #avg, var = value(res)

    # terme de regularisation L2
    #L2 = @views N.λ/(2*n)*sum(sum(N.weights[i][:, 1:end-1].^2) for i = 1:N.num_layers-1)

    # resultat :
    #   - value(res) = valeur cross-entropy avec distribution empirique
    return value(res)
end

#------------------     feedforward!        ---------------------------------

# -- N : Multi Layer Perceptron
# -- β : vecteur des parametres
# -- X : matrice des vecteurs entree = (x_1 | x_2 | ... | x_n)
#           - # lignes = # vecteur entree
#           - # colonne = taille echantillon
#           - X[:,j] = vecteur entree sous individu j
function feedforward!(N::Network, β::AbstractArray, X::AbstractArray)
    for i = 1:N.num_layers-1
        N.weights[i][:,:] = @views reshape(β[N.index[i]], size(N.weights[i]))
        # on remet la partie du vecteur parame/ tres correspondant passage i a i+1
        # sous forme de matrice en remplissant colonne par colonne de haut en bas
    end
    n = size(X, 2) # n = nombre individus dans echantillon
    N.a[1][1:end-1, 1:n] = copy(X) # on initialise premiere couche avec vecteurs entree
    for l = 1:N.num_layers-2
        # ReLU(W*a(l)) pour couches cachees
        # 1:end-1 car derniere colonne doit rester (1, ... ,1)^T pour ajout des biais
        N.a[l+1][1:end-1, 1:n] = @views ReLU(N.weights[l]*N.a[l][:, 1:n])
    end
    # Calcul des probabilites de sortie avec Softmax
    N.a[end][:, 1:n] = @views softmax(N.weights[end]*N.a[end-1][:, 1:n])
end

#-----------------------         backpropagation!        -------------------------

# -- N : Multi Layer Perceptron
# -- β : vecteur des parametres
# -- X : matrice des vecteurs entree = (x_1 | x_2 | ... | x_n)
# -- Y : Matrice des resultats de sortie attendus (y_1 | y_2 | ... | y_n)
# -- gradient : --> output --> Cacul du gradient complet
function backpropagation!(N::Network, β::AbstractArray, X::AbstractArray, Y::AbstractArray, gradient::Vector{Float64})
    n = size(Y, 2) # taille echantillon
    feedforward!(N, β, X) # calcul des probas de sortie avec softMax
    N.δ[1:N.sizes[end], 1:n] = @view N.a[end][:, 1:n] # represente gradient de -ln(P(j|x_i, β))
                                                    # suivant valeur activite derniere couche
    # à voir comme : ∇f_i(β) w.r.t a(L_end) où L_end est la derniere couche d'activite avec ReLU
    for i = 1:n # pour chaque individu de la population
        y = @view Y[:, i]
        j = argmax(y)
        N.δ[j, i] -= 1 # 1 est la proba attendue pour le bon label
    end
    # le Vec est en deux parties :
    #   - une pour derivees partielles w.r.t poids,
    #   - la deuxieme w.r.t biais
    # N.index[end] : indice des parametres derniere couche
    gradient[N.index[end]] = @views vec( [ (N.δ[1:N.sizes[end], 1:n]*(N.a[end-1][1:end-1, 1:n])'/n + N.λ/n*N.weights[end][:, 1:end-1]) mean(eachcol(N.δ[1:N.sizes[end],1:n])) ] )

    for l = (N.num_layers-2):-1:1 # Backpropagation
        # le map correspond a la derivee de ReLU
        N.δ[1:N.sizes[l+1], 1:n] = @views ((N.weights[l+1][:, 1:end-1])'*N.δ[1:N.sizes[l+2], 1:n]).*map(z -> z==0 ? 0 : 1, N.a[l+1][1:end-1, 1:n])
        gradient[N.index[l]] = @views vec([(N.δ[1:N.sizes[l+1], 1:n]*(N.a[l][1:end-1, 1:n])'/n + N.λ/n*N.weights[l][:, 1:end-1]) mean(eachcol(N.δ[1:N.sizes[l+1], 1:n]))])
    end
end

#------------------      backpropagation!       -----------------------------
# idem que precedemment mais on conserve les scores par individus

# -- N : Multi Layer Perceptron
# -- β : vecteur des parametres
# -- X : matrice des vecteurs entree = (x_1 | x_2 | ... | x_n)
# -- Y : Matrice des resultats de sortie attendus (y_1 | y_2 | ... | y_n)
# -- score : ensemble des scores
#               - score[i] correspond au gradient pour individu i
function backpropagation!(N::Network, β::AbstractArray, X::AbstractArray, Y::AbstractArray, score::Array{Vector{Float64}})
    n = size(Y, 2) # taille echantillon
    feedforward!(N, β, X) # calcul des probas de sortie

    # calcul du gradient w.r.t a(L_end)
    N.δ[1:N.sizes[end], 1:n] = @view N.a[end][:, 1:n]
    for i = 1:n
        y = @view Y[:, i]
        j = argmax(y)
        N.δ[j, i] -= 1

        # le Vec est en deux parties :
        #   - une pour derivees partielles w.r.t poids,
        #   - la deuxieme w.r.t biais
        score[i][N.index[end]] = @views vec([(N.δ[1:N.sizes[end], i]*(N.a[end-1][1:end-1, i])' + N.λ/n*N.weights[end][:, 1:end-1])   N.δ[1:N.sizes[end], i]])
        for l = (N.num_layers-2):-1:1
            N.δ[1:N.sizes[l+1], i] = @views ((N.weights[l+1][:, 1:end-1])'*N.δ[1:N.sizes[l+2], i]).*map(z -> z==0 ? 0 : 1, N.a[l+1][1:end-1, i])
            score[i][N.index[l]] = @views vec([(N.δ[1:N.sizes[l+1], i]*(N.a[l][1:end-1, i])' + N.λ/n*N.weights[l][:, 1:end-1]) N.δ[1:N.sizes[l+1], i]])
        end
    end
end

#---------------------       backpropagation!       --------------------------

# -- N : Multi Layer Perceptron
# -- β : vecteur des parametres
# -- X : matrice des vecteurs entree = (x_1 | x_2 | ... | x_n)
# -- Y : Matrice des resultats de sortie attendus (y_1 | y_2 | ... | y_n)

# -- hessian : ouput --> approximation BHHH
function backpropagation!(N::Network, β::AbstractArray, X::AbstractArray, Y::AbstractArray, hessian::Matrix)
    hessian[:, :] = zeros(N.index[end][end], N.index[end][end]) # initialisation Hessienne
    n = size(Y, 2) # taille echantillon
    feedforward!(N, β, X) # calcul probas des labels

    # calcul du gradient w.r.t a(L_end)
    N.δ[1:N.sizes[end], 1:n] = @view N.a[end][:, 1:n]
    for i = 1:n
        score = zeros(N.index[end][end])
        y = @view Y[:, i]
        j = argmax(y)
        N.δ[j, i] -= 1

        # Calcul du score pour individu i
        score[N.index[end]] = @views vec([(N.δ[1:N.sizes[end], i]*(N.a[end-1][1:end-1, i])' + N.λ/n*N.weights[end][:, 1:end-1])   N.δ[1:N.sizes[end], i]])
        for l = (N.num_layers-2):-1:1
            N.δ[1:N.sizes[l+1], i] = @views ((N.weights[l+1][:, 1:end-1])'*N.δ[1:N.sizes[l+2], i]).*map(z -> z==0 ? 0 : 1, N.a[l+1][1:end-1, i])
            score[N.index[l]] = @views vec([(N.δ[1:N.sizes[l+1], i]*(N.a[l][1:end-1, i])' + N.λ/n*N.weights[l][:, 1:end-1]) N.δ[1:N.sizes[l+1], i]])
        end
        hessian[:, :] = @views hessian[:, :] + score*score'
    end
    hessian[:, :] = @views hessian[:, :]/n
end

#---------------------      Sigmoide        --------------------------------
# inp : Input array
sigmoid(inp::AbstractArray) = map(z -> 1.0/(1.0+exp(-z)), inp)

#--------------------     ReLU      -------------------------------------
# inp : input Array
ReLU(inp::AbstractArray) = map(z -> max(z,0), inp)

#---------------------      SoftMax         ----------------------------------
# inp : input array
function softmax(inp::AbstractArray)
    out = similar(inp)
    for (i,v) in enumerate(eachcol(inp))
        m = maximum(v)
        out[:, i] = map(z -> exp(z-m), v)
        out[:, i] = out[:, i]/sum(out[:, i])
    end
    return out
end
