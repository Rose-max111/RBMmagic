module RBMmagic
using LinearAlgebra
export RBM_H_State, RBM_flexable
export sampling
export MetroPolisLocal
export stochastic_reconfiguration_H!

include("rbm.jl")
include("mcmc.jl")
include("optimize.jl")

end