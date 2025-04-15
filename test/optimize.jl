using RBMmagic
using Test
using Random

using RBMmagic:mean

function exact_fidelity(psi::Vector{ComplexF64}, phi::Vector{ComplexF64})
    return (sum(conj.(psi) .* phi) / sqrt(sum(abs2.(psi))) / sqrt(sum(abs2.(phi))))
end

@testset "mean" begin
    a = LinearIndices((3, 4))
    b1 = mean(a, 1)
    @test size(b1, 1) == 4
    @test b1 == [2, 5, 8, 11]
end

@testset "fidelity_grad" begin
    phi_psi = [0.5-2im; 1.5+1im]
    psi_psi = [-0.5-1im; 2.0+3im]
    O = [
        [0.5-2im 1.5+1im 0.5+2im 1.0 -1.0 -0.3;
        -0.5-1im 2.0+3im 0.2+2im 2.0 -2.0 7-1im],
        [0.5-2im 1.5+1im;
        -0.5-1im 2.0+3im],
        [1.0 -1.0;
        2.0 -2.0]
    ]
    @test size(O[1]) == (2, 6)
    @test size(O[2]) == (2, 2)
    @test size(O[3]) == (2, 2)

    grad = RBMmagic.fidelity_grad(phi_psi, psi_psi, O)
    @test size(grad[1]) == (6, )
    @test size(grad[2]) == (2, )
    @test size(grad[3]) == (2, )

    term2_down = mean(exp.(phi_psi - psi_psi))
    # @show term2_down
    # @show grad[1]
    @test grad[1][1] ≈ (1.5im) - ((0.5+2im)*exp((0.5-2im)-(-0.5-1im)) + (-0.5+1im)*exp((1.5+1im)-(2.0+3im))) / term2_down / 2
    @test grad[2] ≈ [-0.22259535-0.51348882im, 0.4749145-0.66336081im]
    @test grad[3] ≈ [0.36804209+0.14544673im, -0.36804209-0.14544673im]
end

@testset "S_matrix" begin
    O = [
        [0.5-2im 1.5+1im 0.5+2im 1.0 -1.0+3im -0.3;
        -0.5-1im 2.0+3im 0.2+2im 2.0 -2.0-1im 7-1im],
        [0.5-2im 1.5+1im;
        -0.5-1im 2.0+3im],
        [1.0-2im -1.0+2im;
        2.0+3im -2.0-1im]
    ]
    S = RBMmagic.S_matrix(O)
    t1 = ((-1.0-3im)*(1.0-2im) + (-2.0+1im)*(2.0+3im))/2
    t2 = (-1.0-3im+-2.0+1im)/2
    t3 = (1.0-2im+2.0+3im)/2
    @test t1-t2*t3 ≈ S[5, 9]
end

@testset "sr_H_update" begin
    O = [
        [0.5-2im 1.5+1im 0.5+2im 1.0 -1.0+3im -0.3;
        -0.5-1im 2.0+3im 0.2+2im 2.0 -2.0-1im 7-1im],
        [0.5-2im 1.5+1im;
        -0.5-1im 2.0+3im],
        [1.0-2im -1.0+2im;
        2.0+3im -2.0-1im]
    ]
    phi_psi = [0.5-2im; 1.5+1im]
    psi_psi = [-0.5-1im; 2.0+3im]
    psi_phi = [0.3-1im; 0.7+2im]
    phi_phi = [0.1-2im; 0.9+3im]

    F, grad_params = RBMmagic.sr_H_update(psi_phi, phi_phi, psi_psi, phi_psi, O, 1e-3)
    @test F ≈ 0.5756537934815336  
    @test isapprox(grad_params[1], [-0.00424003-0.00978101im, 0.00904624-0.01263579im, -0.00210315-0.00083115im, 0.00701052+0.00277049im, -0.01809247+0.02527158im,  0.04840628+0.02723509im];rtol=1e-6)
    @test isapprox(grad_params[2], [-0.00424003-0.00978101im, 0.00904624-0.01263579im]; rtol=1e-6)
    @test isapprox(grad_params[3], [0.02086296-0.03228209im, -0.01532199+0.01826106im]; rtol=1e-6)
end

@testset "mcmc_fidelity" begin
    Random.seed!(1222)
    n=10
    hqubit = 6
    model = RBM_flexable(n, n)
    h_model = RBM_H_State(model, hqubit)
    
    sampler_psi = MetroPolisLocal(model, 4)
    sampler_phi = MetroPolisLocal(h_model, 4)

    samples_psi = sampling(sampler_psi, model; n_samples=2^13*4)
    samples_phi = sampling(sampler_phi, h_model; n_samples=2^13*4)
    @show size(samples_psi)
    @show size(samples_phi)
    psi_psi = lnpsi(model, samples_psi)
    phi_psi = lnpsi(h_model, samples_psi)
    psi_phi = lnpsi(model, samples_phi)
    phi_phi = lnpsi(h_model, samples_phi)
    f_approx = RBMmagic.fidelity(psi_phi, phi_phi, psi_psi, phi_psi)
    @info "fidelity_approx: $f_approx"

    all_state = Matrix(transpose(hcat(collect(reverse.(digits.(0:2^n-1, base=2, pad=n)))...)))
    amplitude_psi = exp.(lnpsi(model, all_state))
    amplitude_phi = exp.(lnpsi(h_model, all_state))
    f_exact = exact_fidelity(amplitude_psi, amplitude_phi)^2
    @info "fidelity_exact: $f_exact"
end

@testset "stochastic_reconfiguration_H" begin
    n = 10
    hqubit = 6
    model = RBM_flexable(n, n)
    h_model = RBM_H_State(model, hqubit)

    @time stochastic_reconfiguration_H!(model, hqubit;
    n_samples=2^13, n_chain_per_rank=8, resample_phi=5)

    all_state = Matrix(transpose(hcat(collect(reverse.(digits.(0:2^n-1, base=2, pad=n)))...)))
    all_amplitude_exact = exp.(lnpsi(h_model, all_state))
    all_amplitude_approx = exp.(lnpsi(model, all_state))
    # aberr = sum(abs.(all_amplitude_approx .- all_amplitude_exact))
    # @show aberr
    @test 1 - exact_fidelity(all_amplitude_approx, all_amplitude_exact) <= 1e-3
end