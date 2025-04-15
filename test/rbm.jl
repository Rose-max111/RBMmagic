using RBMmagic
using Test

@testset "RBM_H_State" begin
    origin_model = RBMmagic.RBM_flexable(3, 4)
    h_model = RBMmagic.RBM_H_State(origin_model, 2)
    
    pre = h_model.origin_state.kernel[1, 2]
    origin_model.kernel[1, 2] = 0.5
    @test h_model.origin_state.kernel[1, 2] != 0.5
    @test h_model.origin_state.kernel[1, 2] == pre
end