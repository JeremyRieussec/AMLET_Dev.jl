
## Ensemble des differentes methodes d'optimisation :

#   -- BTR RA : Basic trust Region avec Retrospective Approximation (utilisant HOPS)
#   -- BTR HOPS : Basic Trust Region avec Hessian-free Outer Product of Scores
#   -- BTR BHHH : Basic Trust Region avec BHHH
#   -- BTR TH : Basic Trust Region avec true Hessian
#   -- BTR BFGS : Basic Trust Region avec BFGS
#   -- AG : Aggregated Gradient
#   -- BFGS


## ------------- Resolution avec Trust-Region Retrospective approximation --------

#                     Seul HOPS est utilise pour BTR_RA

# -- m : type modele
# -- β0 : vecteur parametres
# -- N0 : sample size
# -- coeff : ceffiecient de croissance geometrique pour Sample Size
# -- sample_coeff : SubSampling coeff - pourcentage Sample size
# -- criterion : critere d'arret (cf report Loic p.16/17)
#					- 1 : n_max + first-order robust test sur gradient
#					- 2 :  n_max , robust gradient , tTest de suffiscient decrease
#					- 3 : n_max , robust gradient , over-fitting test --> cross validation
#								--> 50 000 images trainig set
#								--> 10 000 images test set
#								--> 10 000 images validation test
#								--> on arrete quand f(x) sur validation test augmente
# -- verbose : tracage
function solve_BTR_RA(m::Models, β0::AbstractArray, N0::Int64, coeff::Float64;
    sample_coeff::Float64 = 1.0, criterion::Int64 = 2, verbose::Bool = false)
    f_values::Array = []
    nb_iter::Int64 = 0
    m.sample_size = N0
    m.subsample = Int64(round(sample_coeff*m.sample_size))
    m.q_student = quantile(TDist(m.sample_size-1), 1-m.α)
    while m.sample_size <= m.n_train
        if verbose
            println("\n$(nb_iter+=1) : N = $(m.sample_size)\n")
        end
        if m.sample_size < m.n_train || criterion!=3
            if m.sample_size == m.n_train # echantillon total
                m.ϵ = 0.0 # precision maximale demandee
            end
            state, acc = OPTIM_btr_HOPS(m.f, m.score!, β0, m.batch.weights[1:m.subsample], verbose = verbose, epsilon = 1e-20, tTest = m.tTest)
        else
             state, acc = OPTIM_btr_HOPS(m.f_train, m.score!, β0, m.batch.weights[1:m.subsample], verbose = verbose, epsilon = 1e-20, tTest = m.validation)
        end
        β0 = copy(state.x) # mise a jour vecteur parametre
        f_values = copy([f_values; acc]) # conservation des f(x) parcourus
        m.sample_size = Int64(round(m.sample_size*coeff)) # mise a jour sample size
        m.subsample = Int64(round(sample_coeff*m.sample_size)) # maj subsample size
        m.q_student = quantile(TDist(m.sample_size-1), 1-m.α) # maj quantile pour tTest
    end
    # resultats :
    #   - β0 : vecteur optimal
    #   - f_values : ensemble f(x) parcourus
    return β0, f_values
end

#----------------------     Resolution par TR HOPS      ---------------------
# -- idem pour arguments
function solve_BTR_HOPS(m::Models; x0::AbstractArray = zeros(m.dim), verbose::Bool = false, nmax::Int64 = 1000, criterion::Int64 = 1)
    w = Int64[] # vecteur nombre parametres par individus
    for ind in m.batch
        push!(w, ind.n_sim)
    end
    if criterion == 1
        return OPTIM_btr_HOPS(m.f, m.score!, x0, w, verbose = verbose, nmax = nmax)
    end
    if criterion == 2
        return OPTIM_btr_HOPS(m.f, m.score!, x0, w, verbose = verbose, nmax = nmax, epsilon = 1e-20, tTest = m.tTest)
    end
    if criterion == 3
        return OPTIM_btr_HOPS(m.f, m.score!, x0, w, verbose = verbose, nmax = nmax, epsilon = 1e-20, tTest = m.validation)
    end
end

#-------------------         Resolution par TR BHHH     ------------------------
# -- idem pour arguments
function solve_BTR_BHHH(m::Models; x0::AbstractArray = zeros(m.dim), verbose::Bool = false, nmax::Int64 = 1000, criterion::Int64 = 1)
    if criterion == 1
        return OPTIM_btr_TH(m.f, m.∇f!, m.bhhh!, x0, verbose = verbose, nmax = nmax)
    end
    if criterion == 2
        return OPTIM_btr_TH(m.f, m.∇f!, m.bhhh!, x0, verbose = verbose, nmax = nmax, epsilon = 1e-20, tTest = m.tTest)
    end
    if criterion == 3
        return OPTIM_btr_TH(m.f, m.∇f!, m.bhhh!, x0, verbose = verbose, nmax = nmax, epsilon = 1e-20, tTest = m.validation)
    end
end

#---------------        Resolution par TR BFGS      ----------------------------
# -- idem pour arguments
function solve_BTR_BFGS(m::Models; x0::AbstractArray = zeros(m.dim), verbose::Bool = false, nmax::Int64 = 1000, criterion::Int64 = 1)
    if criterion == 1
        return OPTIM_btr_BFGS(m.f, m.∇f!, x0, verbose = verbose, nmax = nmax)
    end
    if criterion == 2
        return OPTIM_btr_BFGS(m.f, m.∇f!, x0, verbose = verbose, nmax = nmax, epsilon = 1e-20, tTest = m.tTest)
    end
    if criterion == 3
        return OPTIM_btr_BFGS(m.f, m.∇f!, x0, verbose = verbose, nmax = nmax, epsilon = 1e-20, tTest = m.validation)
    end
end

#----------------       Resolution par TR True Hessian         -----------------
# -- idem pour arguments
function solve_BTR_TH(m::Models; x0::AbstractArray = zeros(m.dim), verbose::Bool = false, nmax::Int64 = 1000)
    return OPTIM_btr_TH(m.f, m.∇f!, m.Hf!, x0, verbose = verbose, nmax = nmax)
end

function solve_RSAG(lm::Models; verbose::Bool = false, nmax::Int64 = 1000)
    return OPTIM_AGRESSIVE_RSAG(lm.f, lm.∇f!, lm.batch, lm.f::Function; x0 = zeros(m.dim), L = 1.0, nmax = 500,
        ϵ = 1e-4, verbose = false, n_test = 500, n_optim = 100)
end

#---------------------  Resolution par Aggrated Gradient    --------------------
# -- idem pour arguments
function solve_AG(m::Models; verbose::Bool = false, nmax::Int64 = 1000)
    return OPTIM_AGRESSIVE_AG(m.f, m.∇f!, x0 = zeros(m.dim), L = 1.0, nmax = nmax,
        ϵ = 1e-4, verbose = verbose)
end

#------------------     Resolution par BFGS     ---------------------------
# -- idem pour arguments
function solve_BFGS(m::Models; verbose::Bool = false, nmax::Int64 = 1000)
    return OPTIM_BFGS(m.f, m.∇f!, x0 = zeros(m.dim), nmax = nmax,
        ϵ = 1e-4, verbose = verbose)
end

"""
chose between :

  'solve_BTR_BFGS'

  'solve_BTR_TH'

  'solve_RSAG'

  'solve_AG'

  'solve_BFGS'
"""
solve = solve_BTR_BFGS
