
#-------------- 		one-hot encoding 		------------------------

# -- set : labels de sortie
# -- nb_out : nombre de labels possibles en sortie
# ---> permet de creer les couches de sortie a viser lors de l'apprentissage supervise
function one_hot(set::AbstractArray, nb_out::Int64)
    new_set = zeros(nb_out, length(set))
    for i in 1:length(set)
         new_set[set[i]+1, i] = 1
    return new_set
end

#-----------------		evaluation 		----------------
# Pourcentage succes sur un certain ensemble

# -- N : Multi Layer Perceptron
# -- Y : Matrice des resultats de sortie attendus (y_1 | y_2 | ... | y_n)
#       - # lignes = # sorties
#       - # colonne = # data --> taille echantillon sur population
#       - colonne j : resultats de sortie attendue pour data j
#       - ligne i : resultat sortie i
# -- X : matrice des vecteurs entree = (x_1 | x_2 | ... | x_n)
#           - # lignes = # vecteur entree
#           - # colonne = taille echantillon
#           - X[:,j] = vecteur entree sous individu j
# -- β : vecteur parametres
function evaluation(β::AbstractArray, X::AbstractArray, Y::AbstractArray, N::Network)
""" Evaluates the percentage of success of β over the set """
    feedforward!(N, β, X) # forward pass Network N sous parametres β et entree X
    s = 0 # nombre de succes
    n = size(Y, 2) # taille echantillon
    for i = 1:n
        if argmax(N.a[end][:, i]) == argmax(Y[:, i])
            s += 1
        end
    end
    return 100*s/n # pourcentage succes classification sur echantillon
end

#------------------------- 		Test_MNIST 		------------------------------
# Test sur la sur MNIST (par defaut avec retrospective Approximation)

# -- optim : choisir methode optimisation (tous sous TR)
#				- RA : Retrospective Approximation
#				- HOPS : Hessian-free Outer Product of the Scores
#				- BHHH
#				- BFGS
# -- layers : tailles couches INTERNES reseau --> HIDDEN LAYERS sizes
# -- init_seed
# -- set_seed
# -- criterion : critere d'arret (cf report Loic p.16/17)
#					- 1 : first-order robust test sur gradient
#					- 2 : tTest de suffiscient decrease
#					- 3 : over-fitting test --> cross validation
#								--> 50 000 images trainig set
#								--> 10 000 images test set
#								--> 10 000 images validation test
#								--> on arrete quand f(x) sur validation test augmente
# --verbose : tracage

function Test_MNIST(optim::String = "RA", layers::Array = [], init_seed::Int64 = 1,
	set_seed::Int64 = 0, criterion::Int64 = 2, verbose::Bool = false, vargs ...)
"""
vargs[1] = N0::Int64 (if optim != RA)
vargs[1] = lambda::Float64 (if optim != RA)
vargs[2] = coeff::Float64 (if optim == RA)
vargs[3] = eps::Float64 (if optim == RA)
vargs[4] = sample_coeff::Float64 (if optim == RA)
vargs[5] = lambda::Float64 (if optim == RA)
"""
## -- constitution des ensembles de trainning et de test : cf package MNIST
	# -- X : matrice des vecteurs entree = (x_1 | x_2 | ... | x_n)
	# -- Y : Matrice des resultats des labels attendus (y_1 | y_2 | ... | y_n)
    (train_x, train_y) = MNIST.traindata()  # ensemble training (features , labels) = (X , Y)
	(test_x,  test_y)  = MNIST.testdata() # ensemble test (features , labels) = (X , Y)

    ntest = size(test_y, 1) # taille echantillon test
    ntrain = criterion==3 ? size(train_y, 1)-ntest : size(train_y, 1)
    nvalid = criterion==3 ? ntest : 0
	# si critere 3 (over-fitting test) alors :
	#		- on divise traindata en training set + validation test
	#		- sinon traindata au complet
    ntot = ntrain + nvalid + ntest # normalement un total de 70 000
    sizes = size(layers, 1) != 0 ? [784; layers; 10] : [784, 10]
	# nombre de couches internes (en plus de couche input et couche output)
	lambda = optim!="RA" ? vargs[1] : vargs[5]

## ---------------		creation du reseau de neurone		--------------------
    N = Network(sizes, lambda, ntot, init_seed)

## ------------		Creation du Batch MLP 		---------------
	# creation du batch pour le reseau de neurones
	# BatchMLP(features , labels , weights) !!! attention weights pas ceux du reseau
    b = BatchMLP([MNIST.convert2features(train_x) MNIST.convert2features(test_x)], [one_hot(train_y, 10) one_hot(test_y, 10)], ones(Int64, ntot))

	# creation MODELE : MLP(Batch , dimension)
	# N.index[end][end] donne l'indice du denier parametre = taille vecteur param
	mnist = MLP(b, N.index[end][end])


## --------------- Creation de Permutations pour creation Subbatch ----------
    if set_seed > 0
        mrg_gen = MRG32k3aGen([set_seed for i = 1:6])
        mrg = next_stream(mrg_gen)
        perm = [Int(ceil(ntot*rand(mrg))) for i = 1:ntot]
        list = [true for i = 1:ntot]

		# enumerate(perm)  donne ensemble des paires { (i, perm[i]), i=1 .. ntot }
        for (ind, k) in enumerate(perm)
            if list[k]
                list[k] = false
            else
                i = k==ntot ? k : k+1
                j = k==1 ? k : k-1
                while (i<ntot || j>1) && !list[i] && !list[j]
                    i = i==ntot ? i : i+1
                    j = j==1 ? j : j-1
                end
                perm[ind] = list[i] ? i : j
                list[perm[ind]] = false
            end
        end
    else
        perm = [i for i = 1:ntot]
    end

## ------------	Initialisation des batch apres shuffling des indices ------
    mnist.batch.train_x = @view mnist.batch.features[:, perm[1:ntrain]]
    mnist.batch.train_y = @view mnist.batch.labels[:, perm[1:ntrain]]
    mnist.batch.valid_x = @view mnist.batch.features[:, perm[ntrain+1:ntrain+nvalid]]
    mnist.batch.valid_y = @view mnist.batch.labels[:, perm[ntrain+1:ntrain+nvalid]]
    mnist.batch.test_x = @view mnist.batch.features[:, perm[ntrain+nvalid+1:ntot]]
    mnist.batch.test_y = @view mnist.batch.labels[:, perm[ntrain+nvalid+1:ntot]]

    mnist.n_train = ntrain
    mnist.n_valid = nvalid
    mnist.α = 0.05
    mnist.ϵ = 0
    mnist.sample_size = ntrain
    mnist.subsample = mnist.sample_size
    mnist.q_student = quantile(TDist(mnist.sample_size-1), 1-mnist.α)

## -----------------		tTest 		--------------------------------
# -----> renvoie true si on a sufficient decrease
    function tTest(state::GERALDINE.BTRState)
    """ Test if the difference of the two last value of the loss function is less or equal than ϵ*f_old """
        if state.iter != 0 && mnist.f_old != state.fx
            σ2 = mnist.var + mnist.old_var - 2*mnist.cov # cf report Loic p.16
            tstat = (mnist.f_old*(1 - mnist.ϵ) - state.fx)/sqrt(σ2/mnist.sample_size)
            if tstat <= mnist.q_student
                return true
            end
        end
        mnist.f_old = state.fx
        return false
    end

##-------------------- 		validation 		-------------------------------
# -------> renvoie true si decroissance fonction objectif sur validation set
	function validation(state::GERALDINE.BTRState)
    """ Test if the current value of the loss function over the validation set is greater than the previous one """
        ftest = mnist.f_valid(state.x)
        if state.iter != 0 && mnist.f_old < ftest
            return true
        end
        mnist.f_old = ftest
        return false
    end

## -------------		f			-------------------------

# -------> calcul la valeur de la fonction objectif sous parametre β sur taining set
    function f(β::AbstractArray)
        trainx = @view mnist.batch.train_x[:, 1:mnist.sample_size] # copie des entrees
        trainy = @view mnist.batch.train_y[:, 1:mnist.sample_size] # copie des labels en sortie
        feedforward!(N, β, trainx) # forward pass dans reseau de neurone
								   # permet d'avoir les probabilites des labels en sortie
        fvalue, vari, cova = cross_entropy!(N, trainy, mnist.data) # calcul la fonction objectif
		# ------> ici une cross-entropy

		# Mise a jour des attributs utiles au tTest (cf p.16 report Loic)
        mnist.old_var = mnist.var
        mnist.var = vari
        mnist.cov = cova - mnist.f_old*fvalue

		# resultat : valeur fonction objectif
        return fvalue
    end

## ------------------		f_valid			------------------------
# -------> valeur fonction objectif sous β sur validation set
    function f_valid(β::AbstractArray)
        feedforward!(N, β, mnist.batch.valid_x)
        return cross_entropy!(N, mnist.batch.valid_y)
    end

## ------------------		f_valid			------------------------
# -------> valeur fonction objectif sous β sur training set
    function f_train(β::AbstractArray)
        trainx = @view mnist.batch.train_x[:, 1:mnist.sample_size]
        trainy = @view mnist.batch.train_y[:, 1:mnist.sample_size]
        feedforward!(N, β, trainx)
        return cross_entropy!(N, trainy)
    end

## ------------------		g_score!			------------------------
# -------> calcul de l'ensemble des scores sous parametre β
    function g_score!(β::AbstractArray, gradient::Vector{Float64}, score::Array{Vector{Float64}})
        trainx = @view mnist.batch.train_x[:, 1:mnist.sample_size]
        trainy = @view mnist.batch.train_y[:, 1:mnist.sample_size]
        backpropagation!(N, β, trainx, trainy, gradient)
        trainx = @view mnist.batch.train_x[:, 1:mnist.subsample]
        trainy = @view mnist.batch.train_y[:, 1:mnist.subsample]
        backpropagation!(N, β, trainx, trainy, score)
    end

## ------------------		g!			------------------------
# -------> calcul du gradient sous parametre β
    function g!(β::AbstractArray, gradient::Vector{Float64})
        trainx = @view mnist.batch.train_x[:, 1:mnist.sample_size]
        trainy = @view mnist.batch.train_y[:, 1:mnist.sample_size]
        backpropagation!(N, β, trainx, trainy, gradient)
    end

## ------------------		bhhh!			------------------------
# -------> calcul de l'approximation bhhh
    function bhhh!(β::AbstractArray, hessian::Matrix)
        trainx = @view mnist.batch.train_x[:, 1:mnist.sample_size]
        trainy = @view mnist.batch.train_y[:, 1:mnist.sample_size]
        backpropagation!(N, β, trainx, trainy, hessian)
    end

## -------------------		Initialisation avant lancement 		----------------
    mnist.f = f
    mnist.f_valid = f_valid
    mnist.f_train = f_train
    mnist.tTest = tTest
    mnist.validation = validation

	# initialisation du vecteur de parametres initial avec poids
    β0 = Array{Float64}(undef, mnist.dim)
    for i in 1:N.num_layers-1
       β0[N.index[i]] = vec(N.weights[i])
    end

## ------------------		LANCEMENT OPTIIMISATION 		--------------------
# -- vargs[1] = N0::Int64 (if optim == RA)
# -- vargs[1] = lambda::Float64 (if optim != RA)
# -- vargs[2] = coeff::Float64 (if optim == RA)
# -- vargs[3] = eps::Float64 (if optim == RA)
# -- vargs[4] = sample_coeff::Float64 (if optim == RA)
# -- vargs[5] = lambda::Float64 (if optim == RA)

    if optim == "RA"
        if criterion == 1
            error("criterion 1 is not available for RA")
        end
        mnist.ϵ = vargs[3]
        mnist.score! = g_score!
        param, acc = solve_BTR_RA(mnist, β0, vargs[1], vargs[2], sample_coeff = vargs[4], criterion = criterion, verbose = verbose)
    elseif optim == "HOPS"
        mnist.score! = g_score!
        state, acc = solve_BTR_HOPS(mnist, x0 = β0, verbose = verbose, criterion = criterion)
        param = state.x
    elseif optim == "BFGS"
        mnist.∇f! = g!
        state, acc = solve_BTR_BFGS(mnist, x0 = β0, verbose = verbose, criterion = criterion)
        param = state.x
    elseif optim == "BHHH"
        mnist.∇f! = g!
        mnist.bhhh! = bhhh!
        state, acc = solve_BTR_BHHH(mnist, x0 = β0, verbose = verbose, criterion = criterion)
        param = state.x
    else
        error("'", optim,"' does not exist")
        return
    end

    return param, acc, N, mnist
end
# ------------------- END Test_MNIST -------------------------------
##
