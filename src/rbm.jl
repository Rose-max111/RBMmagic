abstract type NeuralNetworkState end

struct RBM_flexable <: NeuralNetworkState
    kernel::Matrix{ComplexF64}
    bias::Vector{ComplexF64}
    local_bias::Vector{ComplexF64}
    function RBM_flexable(kernel::Matrix{ComplexF64}, bias::Vector{ComplexF64}, local_bias::Vector{ComplexF64})
        new(kernel, bias, local_bias)
    end
end
function RBM_flexable(in_features::Int, out_features::Int)
    kernel = 0.01 .* randn(Complex{Float64}, in_features, out_features)
    bias = 0.01 .* randn(Complex{Float64}, out_features)
    local_bias = 0.01 .* randn(Complex{Float64}, in_features)
    return RBM_flexable(kernel, bias, local_bias)
end
function lnpsi(state::RBM_flexable, x::Vector{Int}) # x: (nfeatures,)
    y = (transpose(x) * state.kernel + transpose(state.bias))
    return sum(log.(2*cosh.(y))) + transpose(x) * state.local_bias
end
function lnpsi(state::RBM_flexable, x::Matrix{Int}) # x: (nsamples, nfeatures)
    y = (x * state.kernel .+ transpose(state.bias))
    return vec(sum(log.(2*cosh.(y)), dims=2) .+ x * state.local_bias)
end
function control_z!(state::RBM_flexable, ctrl_qubit::Int, target_qubit::Int)
    appended_kernel = zeros(ComplexF64, 1, size(state.kernel, 2))
    appended_kernel[1, ctrl_qubit] = -1im * pi / 3
    appended_kernel[1, target_qubit] = 1im * arctan(2 / sqrt(3))
    state.kernel = vcat(state.kernel, appended_kernel)

    state.bias = vcat(state.bias, 1im * pi / 3)

    state.local_bias[ctrl_qubit] -= log(2)
    state.local_bias[target_qubit] += 1 / 2 * log(7/3) + 1im * pi
end

struct RBM_H_State <: NeuralNetworkState
    origin_state::RBM_flexable
    qubit::Int
    function RBM_H_State(origin_state::RBM_flexable, qubit::Int)
        new(deepcopy(origin_state), qubit)
    end
end
function lnpsi(state::RBM_H_State, x::Vector{Int}) # x: (nfeatures,)
    v1 = lnpsi(state.origin_state, x)
    v2 = lnpsi(state.origin_state, [x[1:state.qubit-1]..., 1-x[state.qubit], x[state.qubit+1:end]...])
    return log((exp(v2) + exp(v1) * (1 - 2*x[state.qubit])) / sqrt(2))
end
function lnpsi(state::RBM_H_State, x::Matrix{Int}) # x: (nsamples, nfeatures)
    v1 = lnpsi(state.origin_state, x)
    v2 = lnpsi(state.origin_state, cat(x[:, 1:state.qubit-1], 1 .- x[:, state.qubit], x[:, state.qubit+1:end], dims=2))
    return vec(log.((exp.(v2) .+ exp.(v1) .* (1 .- 2 .* x[:, state.qubit])) ./ sqrt(2)))
end

function log_wf_grad(state::RBM_flexable, x::Matrix{Int})
    local_bias_grad = deepcopy(x)
    bias_grad = tanh.(x * state.kernel .+ transpose(state.bias))
    kernel_grad = reshape(x, (size(x)..., 1)) .* reshape(bias_grad, (size(bias_grad, 1), 1, :))
    return [reshape(kernel_grad, size(kernel_grad, 1), :), bias_grad, local_bias_grad]
end
