abstract type AbstractSampler end

struct MetroPolisLocal <: AbstractSampler
    n_sweeps::Int
    n_chains::Int
    function MetroPolisLocal(n_sweeps::Int, n_chains::Int)
        new(n_sweeps, n_chains)
    end
end
MetroPolisLocal(state::RBM_flexable) = MetroPolisLocal(size(state.kernel, 1), 1) # default nsweep is the number of spins, one chain
MetroPolisLocal(state::RBM_flexable, n_chains::Int) = MetroPolisLocal(size(state.kernel, 1), n_chains) # default nsweep is the number of spins
MetroPolisLocal(state::RBM_H_State) = MetroPolisLocal(size(state.origin_state.kernel, 1), 1) # default nsweep is the number of spins, one chain
MetroPolisLocal(state::RBM_H_State, n_chains::Int) = MetroPolisLocal(size(state.origin_state.kernel, 1), n_chains) # default nsweep is the number of spins


function single_sample(sampler::MetroPolisLocal, nnstate::RBM_flexable, basis::Vector{Int}, res::Vector{ComplexF64}, lnpsire::Float64)
    # Generate a sample using the Metropolis local sampler
    local_bias_sum = sum(basis .* nnstate.local_bias)
    for _ in 1:sampler.n_sweeps
        # Randomly select a spin to flip
        i = rand(1:size(nnstate.kernel, 1))
        
        delta_res = (1-2*basis[i]) .* nnstate.kernel[i, :]
        delta_local_bias_sum = (1-2*basis[i]) .* nnstate.local_bias[i]
        # newpsi = sum(log.(2*cosh.(res .+ delta_res))) + sum([basis[1:i-1]..., basis[i]⊻1, basis[i+1:end]...] .* nnstate.local_bias)
        newpsi = sum(log.(2*cosh.(res .+ delta_res))) + local_bias_sum + delta_local_bias_sum
        delta = 2*(real(newpsi) - lnpsire)
        # Accept or reject the flip based on the Metropolis criterion
        if rand() < exp(min(delta, 0))
            basis[i] = basis[i] ⊻ 1
            lnpsire = real(newpsi)
            res = res + delta_res
            local_bias_sum += delta_local_bias_sum
        end
    end
    return basis, res, lnpsire
end

function single_sample(sampler::MetroPolisLocal, nnstate::RBM_H_State, basis::Vector{Int}, res::Vector{ComplexF64}, lnpsire::Float64)
    # H_state = ln(± exp(lnpsi(basis)) + exp(lnpsi(basis ⊻ 2^nnstate.qubit)))
    local_bias_sum1 = sum(basis .* nnstate.origin_state.local_bias) # calculate lnpsi(basis)
    local_bias_sum2 = sum([basis[1:nnstate.qubit-1]..., 1-basis[nnstate.qubit], basis[nnstate.qubit+1:end]...] .* nnstate.origin_state.local_bias) #calculate lnpsi(basis ⊻ 2^nnstate.qubit)
    for _ in 1:sampler.n_sweeps
        i = rand(1:size(nnstate.origin_state.kernel, 1))
        delta_res1 = (1-2*basis[i]) .* nnstate.origin_state.kernel[i, :]
        delta_res2 = ((i==nnstate.qubit) ? (2*basis[i]-1) : (1-2*basis[i])) .* nnstate.origin_state.kernel[i, :]
        delta_local_bias_sum1 = (1-2*basis[i]) .* nnstate.origin_state.local_bias[i]
        delta_local_bias_sum2 = ((i==nnstate.qubit) ? (2*basis[i]-1) : (1-2*basis[i])) .* nnstate.origin_state.local_bias[i]
        newpsi1 = sum(log.(2*cosh.(res[1:size(nnstate.origin_state.kernel, 2)] .+ delta_res1))) + local_bias_sum1 + delta_local_bias_sum1
        newpsi2 = sum(log.(2*cosh.(res[size(nnstate.origin_state.kernel, 2)+1:end] .+ delta_res2))) + local_bias_sum2 + delta_local_bias_sum2
        
        if i == nnstate.qubit
            if basis[i] == 1 # this time is 1, then flip = 0
                newpsi = log((exp(newpsi1) + exp(newpsi2))/sqrt(2))
            else
                newpsi = log((exp(newpsi2) - exp(newpsi1))/sqrt(2))
            end
        else
            newpsi = log((exp(newpsi2) + exp(newpsi1) * (1 - 2*basis[nnstate.qubit]))/sqrt(2))
        end
        delta = 2*(real(newpsi) - lnpsire)
        if rand() < exp(min(delta, 0))
            basis[i] = basis[i] ⊻ 1
            lnpsire = real(newpsi)
            res = res .+ cat(delta_res1, delta_res2, dims=1)
            local_bias_sum1 += delta_local_bias_sum1
            local_bias_sum2 += delta_local_bias_sum2
        end
        return basis, res, lnpsire
    end
end

function preparebasis(nnstate::RBM_flexable, n_chains::Int)
    return rand([0, 1], n_chains, size(nnstate.kernel, 1))
end
function preparebasis(nnstate::RBM_H_State, n_chains::Int)
    return rand([0, 1], n_chains, size(nnstate.origin_state.kernel, 1))
end
function prepareres(nnstate::RBM_flexable, basis::Matrix{Int})
    return (basis * nnstate.kernel) .+ transpose(nnstate.bias)
end
function prepareres(nnstate::RBM_H_State, basis::Matrix{Int})
    return cat(basis * nnstate.origin_state.kernel .+ transpose(nnstate.origin_state.bias),
            cat(basis[:, 1:nnstate.qubit-1], 1 .- basis[:, nnstate.qubit], basis[:, nnstate.qubit+1:end], dims=2) * nnstate.origin_state.kernel .+ transpose(nnstate.origin_state.bias),
            dims = 2)
end
function preparesamples(nnstate::RBM_flexable, n_chains::Int, n_samples_per_chain::Int)
    return zeros(Int, n_chains, n_samples_per_chain, size(nnstate.kernel, 1))
end
function preparesamples(nnstate::RBM_H_State, n_chains::Int, n_samples_per_chain::Int)
    return zeros(Int, n_chains, n_samples_per_chain, size(nnstate.origin_state.kernel, 1))
end


function sampling(sampler::MetroPolisLocal, nnstate::NeuralNetworkState; n_samples::Int, n_discard = div(n_samples, 10))
    basis = preparebasis(nnstate, sampler.n_chains)
    res = prepareres(nnstate, basis)
    lnpsire = real.(lnpsi(nnstate, basis))

    n_samples_per_chain = div(n_samples, sampler.n_chains)
    if n_samples_per_chain * sampler.n_chains != n_samples
        @info "n_samples $n_samples is not divisible by n_chains $(sampler.n_chains), raised to $((1+n_samples_per_chain) * sampler.n_chains)"
        n_samples_per_chain += 1
    end
    samples = preparesamples(nnstate, sampler.n_chains, n_samples_per_chain)
    # discard some samples to thermalize
    for _ in 1:n_discard
        for i in 1:sampler.n_chains
            basis[i, :], res[i, :], lnpsire[i] = single_sample(sampler, nnstate, basis[i, :], res[i, :], lnpsire[i])
        end
    end
    for num in 1:n_samples_per_chain
        for i in 1:sampler.n_chains
            basis[i, :], res[i, :], lnpsire[i] = single_sample(sampler, nnstate, basis[i, :], res[i, :], lnpsire[i])
            samples[i, num, :] = basis[i, :]
        end
    end
    if typeof(nnstate) <: RBM_H_State
        ret = reshape(samples, :, size(nnstate.origin_state.kernel, 1))
    else
        ret = reshape(samples, :, size(nnstate.kernel, 1))
    end
    return ret[1:n_samples, :]
end