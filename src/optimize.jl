function logsumexp(x::AbstractVector, b::Float64)
    # log(b*sum(exp(x)))
    max_x = maximum(real(x))
    return max_x + log(sum(exp.(x .- max_x))) + log(b)
end

function mean(x::AbstractMatrix, dims::Int)
    return vec(sum(x, dims=dims) ./ size(x, dims))
end
function mean(x::AbstractVector)
    return sum(x) / length(x)
end

function fidelity(psi_phi::Vector{ComplexF64}, phi_phi::Vector{ComplexF64}, 
    psi_psi::Vector{ComplexF64}, phi_psi::Vector{ComplexF64})
    o1 = logsumexp(phi_psi - psi_psi, 1.0/size(psi_phi, 1))
    o2 = logsumexp(psi_phi - phi_phi, 1.0/size(psi_phi, 1))
    return real(exp(o1 + o2))
end

function fidelity_grad(phi_psi::Vector{ComplexF64}, psi_psi::Vector{ComplexF64}, O::Vector{Matrix{ComplexF64}})
    term1 = map(x->mean(conj.(x), 1), O)
    ratio = exp.(phi_psi - psi_psi)
    term2_down = sum(ratio) / size(ratio, 1)
    term2 = map(x->mean(conj.(x).*ratio, 1) / term2_down, O)
    return term1 - term2
end

function S_matrix(O::Vector{Matrix{ComplexF64}})
    merged_O = cat(O..., dims=2)
    estO = merged_O .- transpose(mean(merged_O, 1))
    return (estO' * estO) ./ size(estO, 1)
end

function sr_H_update(psi_phi::Vector{ComplexF64}, phi_phi::Vector{ComplexF64}, 
    psi_psi::Vector{ComplexF64}, phi_psi::Vector{ComplexF64}, O::Vector{Matrix{ComplexF64}}, eps::Float64)
    F = fidelity(psi_phi, phi_phi, psi_psi, phi_psi)
    grad_F = fidelity_grad(phi_psi, psi_psi, O)

    S = S_matrix(O)
    S += eps * I

    merged_grad_F = cat(grad_F..., dims=1)
    inds = cumsum([1; size.(grad_F, 1)])
    merged_grad_params = S \ merged_grad_F
    grad_params = [merged_grad_params[inds[i]:inds[i+1]-1] for i in 1:length(inds)-1]
    grad_params .*= F
    return F, grad_params
end

function stochastic_reconfiguration_H!(model::RBM_flexable, qubit::Int; tol=1e-3, lookback=5, max_iters=1000, resample_phi=Nothing, lr=1e-1, lr_tau=Nothing,
    lr_nstep=Nothing, lr_min=0.0, eps=1e-4,
    n_samples = 2^13,
    n_chain_per_rank = 8)
    pertubated_kernel = model.kernel + randn(ComplexF64, size(model.kernel)) * 1e-2
    pertubated_bias = model.bias + randn(ComplexF64, size(model.bias)) * 1e-2
    pertubated_local_bias = model.local_bias + randn(ComplexF64, size(model.local_bias)) * 1e-2
    
    target_model = RBM_H_State(model, qubit)
    model.kernel .= pertubated_kernel
    model.bias .= pertubated_bias
    model.local_bias .= pertubated_local_bias
    
    sampler_phi = MetroPolisLocal(target_model, n_chain_per_rank)
    sampler_psi = MetroPolisLocal(model, n_chain_per_rank)
    samples_phi = sampling(sampler_phi, target_model; n_samples=n_samples)
    phi_phi = lnpsi(target_model, samples_phi)

    history = Vector{Float64}()
    F = 0.0
    F_mean_new = 0.0
    F_mean_old = 0.0
    diff_mean_F = 2*tol
    step = 0

    while (diff_mean_F > tol || step < 2*lookback+1) && F_mean_new < 1-tol && step < max_iters
        step += 1
        samples_psi = sampling(sampler_psi, model; n_samples=n_samples)
        psi_phi = lnpsi(model, samples_phi)
        phi_psi = lnpsi(target_model, samples_psi)
        psi_psi = lnpsi(model, samples_psi)
        
        O = log_wf_grad(model, samples_psi)
        F, grad_params = sr_H_update(psi_phi, phi_phi, psi_psi, phi_psi, O, eps)
        append!(history, F)

        model.kernel .-= lr .* reshape(grad_params[1], size(model.kernel))
        model.bias .-= lr .* grad_params[2]
        model.local_bias .-= lr .* grad_params[3]

        if lr_tau !== Nothing && step % lr_nstep == 0
            lr = lr * lr_tau
            lr = max(lr, lr_min)
        end
        
        if step > 2*lookback
            F_mean_new = sum(history[end-lookback+1:end]) / lookback
            F_mean_old = sum(history[end-2*lookback+1:end-lookback]) / lookback
        end
        diff_mean_F = abs(F_mean_new - F_mean_old)
        if resample_phi !== Nothing && step % resample_phi == 0
            samples_phi = sampling(sampler_phi, target_model; n_samples=n_samples)
            phi_phi = lnpsi(target_model, samples_phi)
        end
        @info "Step: $step, F: $F, diff_mean_F: $diff_mean_F, lr: $lr"
    end
end