
#------------------    Modele Structure Multi layer Perceptron        --------------------

mutable struct MLP <: Models
    batch::Batch # Batch de data
    dim::Int64 # dimension du vecteur de parametres
    n_train::Int64 # taille training set
    n_valid::Int64 # taille validation set
    sample_size::Int64 # taille echantillon
    subsample::Int64 # taille sous-echantillon
    q_student::Float64 # (1-α)-quantile pour tTest
    α::Float64 # niveau de confiance
    ϵ::Float64 # facteur de sufficient decrease dans tTest

    ## -- Attributs pour le tTest
    f_old::Float64 # valeur f(β_{k-1})
    old_var::Float64 # variance des erreurs sous parametre β_{k-1}
                     # X_{k-1} = { -ln(P(y_label(xi)|x_i) } i = 1 ... N
    var::Float64 # variance des erreurs sous parametre β_{k}
    cov::Float64 # Covariance des erreurs (k-1) et (k)
                 # cov = Cov(X_{k-1} , X_{k})
    data::Array{Float64} # permet de stocker les erreurs de líteration precedente
                         # X_{k-1} = { -ln(P(y_label(xi)|x_i) } i = 1 ... N

    f::Function # fonction objectif
    f_valid::Function # caclul de la valeur de la fcontion objectif sur validation set
    f_train::Function # caclul de la valeur de la fcontion objectif sur training set
    score!::Function # scores de la fonction objectif
    ∇f!::Function # gradient de f
    bhhh!::Function # approximation BHHH de la hessienne.
    Hf!::Function # True Hessienne de f
    tTest::Function # tTest pour sufficient decrease
    validation::Function # test over-fitting

    # Constructeur MLP
    function MLP(b::Batch, dim::Int64)
        m = new()
        m.dim = dim
        m.batch = b
        n_total = sum(b.weights)
        m.data = zeros(Int(n_total))
        m.var = 0
        m.f_old = 0
        return m
    end
end
