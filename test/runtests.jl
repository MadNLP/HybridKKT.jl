using CUDA
using ExaModels
using HybridKKT
using LinearAlgebra
using MadNLP
using MadNLPGPU
using NLPModels
using Random
using SparseArrays
using Test

# COPS instance from: https://github.com/exanauts/ExaModelsExamples.jl/blob/main/src/elec.jl
function elec_model(np; seed = 2713, T = Float64, backend = nothing, kwargs...)
    Random.seed!(seed)
    # Set the starting point to a quasi-uniform distribution of electrons on a unit sphere
    theta = (2pi) .* rand(np)
    phi = pi .* rand(np)

    core = ExaModels.ExaCore(T; backend=backend)
    x = ExaModels.variable(core, 1:np; start = [cos(theta[i])*sin(phi[i]) for i=1:np])
    y = ExaModels.variable(core, 1:np; start = [sin(theta[i])*sin(phi[i]) for i=1:np])
    z = ExaModels.variable(core, 1:np; start = [cos(phi[i]) for i=1:np])
    # Coulomb potential
    itr = [(i,j) for i in 1:np-1 for j in i+1:np]
    ExaModels.objective(core, 1.0 / sqrt((x[i] - x[j])^2 + (y[i] - y[j])^2 + (z[i] - z[j])^2) for (i,j) in itr)
    # Unit-ball
    ExaModels.constraint(core, x[i]^2 + y[i]^2 + z[i]^2 - 1 for i=1:np)

    return ExaModels.ExaModel(core; kwargs...)
end

function initialize_kkt!(kkt, cb)
    MadNLP.initialize!(kkt)
    # Compute initial values for Hessian and Jacobian
    x0 = NLPModels.get_x0(cb.nlp)
    y0 = NLPModels.get_y0(cb.nlp)
    # Update Jacobian manually
    jac = MadNLP.get_jacobian(kkt)
    MadNLP._eval_jac_wrapper!(cb, x0, jac)
    MadNLP.compress_jacobian!(kkt)
    # Update Hessian manually
    hess = MadNLP.get_hessian(kkt)
    MadNLP._eval_lag_hess_wrapper!(cb, x0, y0, hess)
    MadNLP.compress_hessian!(kkt)

    MadNLP._set_aug_diagonal!(kkt)
    MadNLP.build_kkt!(kkt)
    return
end

function test_hybrid_kkt_cpu(nlp, linear_solver)
    # Callback
    ind_cons = MadNLP.get_index_constraints(
        nlp,
    )
    cb = MadNLP.create_callback(
        MadNLP.SparseCallback,
        nlp,
    )

    # Build reference KKT system (here SparseKKTSystem)
    kkt_ref = MadNLP.create_kkt_system(
        MadNLP.SparseKKTSystem, cb, ind_cons, linear_solver;
    )
    initialize_kkt!(kkt_ref, cb)
    MadNLP.factorize!(kkt_ref.linear_solver)
    x_ref = MadNLP.UnreducedKKTVector(kkt_ref)
    MadNLP.full(x_ref) .= 1.0
    MadNLP.solve!(kkt_ref, x_ref)

    # Build HybridCondensedKKTSystem
    kkt = MadNLP.create_kkt_system(
        HybridKKT.HybridCondensedKKTSystem, cb, ind_cons, linear_solver;
    )
    initialize_kkt!(kkt, cb)
    MadNLP.factorize!(kkt.linear_solver)
    x = MadNLP.UnreducedKKTVector(kkt)
    MadNLP.full(x) .= 1.0
    MadNLP.solve!(kkt, x)

    # Test backsolve returns the same values as with SparseKKTSystem.
    @test x.values ≈ x_ref.values atol=1e-6

    # Test consistency of Jacobian
    n = get_nvar(nlp)
    @test kkt.jt_csc' == kkt_ref.jac_com[:, 1:n]
    @test kkt.jt_csc[:, kkt.ind_eq] == kkt.G_csc'

    # Test KKT multiplication
    b = MadNLP.UnreducedKKTVector(kkt)
    mul!(b, kkt, x)
    @test MadNLP.full(b) ≈ ones(length(b)) atol=1e-6
end

function test_hybrid_kkt_cuda(nlp, linear_solver)
    # Callback
    ind_cons = MadNLP.get_index_constraints(
        nlp,
    )
    cb = MadNLP.create_callback(
        MadNLP.SparseCallback,
        nlp,
    )
    # Build HybridCondensedKKTSystem
    kkt = MadNLP.create_kkt_system(
        HybridKKT.HybridCondensedKKTSystem, cb, ind_cons, linear_solver;
    )
    initialize_kkt!(kkt, cb)
    MadNLP.factorize!(kkt.linear_solver)
    x = MadNLP.UnreducedKKTVector(kkt)
    MadNLP.full(x) .= 1.0
    MadNLP.solve!(kkt, x)
    # Test KKT multiplication
    b = MadNLP.UnreducedKKTVector(kkt)
    # TODO: memory fault when calling mul! on the GPU
    mul!(b, kkt, x)
    bvals = Array(MadNLP.full(b))
    @test bvals ≈ ones(length(b)) atol=1e-6
end

@testset "[CPU] HybridCondensedKKTSystem" begin
    nlp = elec_model(5)
    linear_solver = LapackCPUSolver
    # Test HybridKKTSystem is returning the correct result
    test_hybrid_kkt_cpu(nlp, linear_solver)
end

if CUDA.functional()
    @testset "[GPU] HybridCondensedKKTSystem" begin
        nlp = elec_model(5; backend=CUDABackend())
        linear_solver = MadNLPGPU.LapackGPUSolver
        # Test HybridKKTSystem is returning the correct result
        test_hybrid_kkt_cuda(nlp, linear_solver)
    end
end

@testset "Condensed-space IPM" begin
    # Compute reference solution
    nlp = elec_model(5)
    # Reference
    solver_ref = MadNLPSolver(
        nlp;
        kkt_system=MadNLP.SparseKKTSystem,
        linear_solver=LapackCPUSolver,
        print_level=MadNLP.ERROR,
        tol=1e-8,
    )
    stats_ref = MadNLP.solve!(solver_ref)

    @testset "[CPU] LapackCPUSolver" begin
        solver = MadNLPSolver(
            nlp;
            kkt_system=HybridKKT.HybridCondensedKKTSystem,
            linear_solver=LapackCPUSolver,
            lapack_algorithm=MadNLP.CHOLESKY,
            print_level=MadNLP.ERROR,
            tol=1e-8,
        )
        stats = MadNLP.solve!(solver)
        @test stats.status == MadNLP.SOLVE_SUCCEEDED
        @test stats.iter == stats_ref.iter
        @test stats.solution ≈ stats_ref.solution atol=1e-6
    end

    if CUDA.functional()
        nlp_gpu = elec_model(5; backend=CUDABackend())
        @testset "[CUDA] LapackGPUSolver" begin
            solver = MadNLPSolver(
                nlp_gpu;
                linear_solver=MadNLPGPU.LapackGPUSolver,
                lapack_algorithm=MadNLP.CHOLESKY,
                kkt_system=HybridKKT.HybridCondensedKKTSystem,
                equality_treatment=MadNLP.EnforceEquality,
                fixed_variable_treatment=MadNLP.MakeParameter,
                print_level=MadNLP.ERROR,
                inertia_correction_method=MadNLP.InertiaBased,
                max_iter=200,
                tol=1e-5,
            )
            solver.kkt.gamma[] = 1e7
            stats = MadNLP.solve!(solver)
            @test stats.status == MadNLP.SOLVE_SUCCEEDED
            @test stats.iter == stats_ref.iter
            @test Array(stats.solution) ≈ stats_ref.solution atol=1e-6
        end
        @testset "[CUDA] MadNLPGPU.CUDSSSolver" begin
            solver = MadNLPSolver(
                nlp_gpu;
                linear_solver=MadNLPGPU.CUDSSSolver,
                cudss_algorithm=MadNLP.LDL,
                kkt_system=HybridKKT.HybridCondensedKKTSystem,
                equality_treatment=MadNLP.EnforceEquality,
                fixed_variable_treatment=MadNLP.MakeParameter,
                print_level=MadNLP.ERROR,
                inertia_correction_method=MadNLP.InertiaBased,
                max_iter=200,
                tol=1e-5,
            )
            solver.kkt.gamma[] = 1e7
            stats = MadNLP.solve!(solver)
            @test stats.status == MadNLP.SOLVE_SUCCEEDED
            @test stats.iter == stats_ref.iter
            @test Array(stats.solution) ≈ stats_ref.solution atol=1e-6
        end
    end
end

