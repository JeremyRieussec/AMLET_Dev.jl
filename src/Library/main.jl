#basic function that should be in julia: iterator ovec colum and line of matrx and eye

#Individus contains the abstract declaration of an individus and the declaration for LM_Individus and MLM_Individus.
include("Individual/main.jl")


#Batch contains the abstract declaration for the Bacth Type and a generic iterator.
include("Batch/main.jl")

#Utilities
include("Utilities/main.jl")

#Models contains the abstract declaration for an abstract Model.
include("Models/main.jl")

#####include("Optimize/stop.jl")
#solving methods
#####include("solve/solve.jl")

include("logit/Library_logit/main.jl")

include("Mixed_Logit/main.jl")

include("Tests/main.jl")