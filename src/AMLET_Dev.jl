"""
A version of the well known AMLET package.

See AMLET at https://github.com/JLChartrand/AMLET.jl.git
"""

module AMLET

#The used Package
using RDST, Statistics, LinearAlgebra, Distributions, Random, GERALDINE, OnlineStats, MLDatasets

#function redefined in AMLET
import Base.iterate, Base.copy

export
    # batch types
    Batch, BatchLM, BatchMLM, BatchMLP, iterate,

    #basic function
    iterate, copy,

    #Individus Type
    Individual, LM_Individual, MLM_Individual,

    #Models type
    Models, LM, MLM, MLP,
    #function
    get_cov_matrix, CI,

    #Utilities type
    Utilities, LinearUtilities,

    #predefine utility
    LU, UVINLU,

    #A little surprise for a unpredicted reader
    col, line,

    #the solve function
    #solve,

    #The well known complete_Model! function
    complete_Model!,

    #others
    mean,

    #Solving gears
    solve, solve_BTR_BFGS, solve_BTR_TH, solve_RSAG, solve_AG, solve_BFGS, solve_BTR_HOPS, solve_BTR_BHHH, solve_BTR_RA,

    #Tests
    Network,
    evaluation, Test_MNIST


include("Library/main.jl")



end # module
