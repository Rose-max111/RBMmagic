using RBMmagic
using Test
using RBMmagic: RBM_flexable, sampling, MetroPolisLocal
using RBMmagic: lnpsi
using Random

function vec2int(v)
    return parse(Int, join(v), base=2)
end

@testset "mcmc_rbm" begin
    n = 10
    model = RBM_flexable(zeros(ComplexF64, n, n), ones(ComplexF64, n) .* (1im * pi / 3), [log(2), log(2), 1im * pi / 2, 1im * pi / 2, 1im * pi / 2, log(2), log(2), 1im * pi / 2, 1im * pi / 2, 1im * pi / 2])
    all_state = Matrix(transpose(hcat(collect(reverse.(digits.(0:2^n-1, base=2, pad=n)))...)))
    all_amplitude = exp.(lnpsi(model, all_state))
    for i in 0:2^n-1
        @test all_amplitude[i+1] ≈ exp(lnpsi(model, all_state[i+1, :]))
    end
    
    pdf_unnormalized = conj.(all_amplitude) .* all_amplitude
    pdf = real.(pdf_unnormalized ./ sum(pdf_unnormalized))

    sampler = MetroPolisLocal(model, 16)
    @time samples = sampling(sampler, model; n_samples = 2^13*sampler.n_chains, n_discard = 100)
    samples_int = vec2int.(eachrow(samples))
    pdf_estimate = [count(x -> x == i, samples_int) / length(samples_int) for i in 0:2^n-1]
    # @info pdf_estimate
    # @info pdf
    @info sum(abs.(pdf .- pdf_estimate))
end

@testset "h_state" begin
    Random.seed!(78321)
    n = 10
    cqubit = 2
    model = RBM_flexable(10, 15)
    # model = RBM_flexable(zeros(ComplexF64, n, n), ones(ComplexF64, n) .* (1im * pi / 3), [log(2), log(2), 1im * pi / 2, 1im * pi / 2, 1im * pi / 2, log(2), log(2), 1im * pi / 2, 1im * pi / 2, 1im * pi / 2])
    hmodel = RBM_H_State(model, cqubit)
    all_state = Matrix(transpose(hcat(collect(reverse.(digits.(0:2^n-1, base=2, pad=n)))...)))
    all_amplitude = exp.(lnpsi(hmodel, all_state))
    for i in 0:2^n-1
        @test all_amplitude[i+1] ≈ exp(lnpsi(hmodel, all_state[i+1, :]))
    end
    
    pdf_unnormalized = conj.(all_amplitude) .* all_amplitude
    pdf = real.(pdf_unnormalized ./ sum(pdf_unnormalized))

    sampler = MetroPolisLocal(hmodel, 4)
    @time samples = sampling(sampler, hmodel; n_samples = 2^13*sampler.n_chains, n_discard = 100)
    samples_int = vec2int.(eachrow(samples))
    pdf_estimate = [count(x -> x == i, samples_int) / length(samples_int) for i in 0:2^n-1]
    # @info pdf_estimate
    # @info pdf
    @info sum(abs.(pdf .- pdf_estimate))
end