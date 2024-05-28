# HybridKKT.jl

A [MadNLP](madnlp) implementation of the [Golub & Greif KKT solver](golub2003).
This package provides an `HybridCondensedKKTSystem` structure for MadNLP,
with GPU support.

## Quickstart

We implement using [ExaModels](examodels) the `elec` instance
from the [COPS benchmark](cops). The problem models the distribution
of electrons on a sphere.
```julia
using ExaModels

function elec_model(np; seed = 2713, T = Float64, backend = nothing, kwargs...)
    Random.seed!(seed)
    # Set the starting point to a quasi-uniform distribution of electrons on a unit sphere
    theta = (2pi) .* rand(np)
    phi = pi .* rand(np)

    core = ExaModels.ExaCore(T; backend= backend)
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

```
ExaModels allows to evaluate the model on different backends.


### Solve the instance on the CPU
We instantiate a model on the CPU with 5 electrons as:
```julia
nlp = elec_model(5)
```
To solve the problem with a `HybridCondensedKKTSystem`,
call MadNLP with the following arguments:
```julia
using MadNLP
using HybridKKT

solver = MadNLPSolver(
    nlp;
    kkt_system=HybridKKT.HybridCondensedKKTSystem,
    linear_solver=LapackCPUSolver,
    lapack_algorithm=MadNLP.CHOLESKY,
)
stats = MadNLP.solve!(solver)
```

Increasing the parameter `gamma` can lead to a faster solution, at the
expense of the accuracy:
```julia
solver = MadNLPSolver(
    nlp;
    kkt_system=HybridKKT.HybridCondensedKKTSystem,
    linear_solver=LapackCPUSolver,
    lapack_algorithm=MadNLP.CHOLESKY,
)
solver.kkt.gamma[] = 1e8 # change gamma
stats = MadNLP.solve!(solver)
```

### Solve the instance on the GPU
To solve the problem on the GPU, you need to load the package
MadNLPGPU to have full GPU support in MadNLP. Then, you can instantiate
the problem on the GPU as
```julia
using CUDA

nlp_gpu = elec_model(5; backend=CUDABackend())
```

To solve it using GPU-accelerated solver, call MadNLP with
the following arguments:
```julia
solver = MadNLPSolver(
    nlp_gpu;
    linear_solver=MadNLPGPU.LapackGPUSolver,
    lapack_algorithm=MadNLP.CHOLESKY,
    kkt_system=HybridKKT.HybridCondensedKKTSystem,
    equality_treatment=MadNLP.EnforceEquality,
    fixed_variable_treatment=MadNLP.MakeParameter,
)
stats = MadNLP.solve!(solver)

```

If the problem is too large, you can replace Lapack by the
sparse linear solver [cuDSS](cudss), provided by NVIDIA.
MadNLP uses the Julia interface [CUDSS.jl](cudssjl).
We recommend using the LDL factorization in cuDSS, as the Cholesky
solver does not report correctly whether the factorization has succeeded.
The arguments write:
```julia
solver = MadNLPSolver(
    nlp_gpu;
    linear_solver=MadNLPGPU.CUDSSSolver,
    cudss_algorithm=MadNLP.LDL,
    kkt_system=HybridKKT.HybridCondensedKKTSystem,
    equality_treatment=MadNLP.EnforceEquality,
    fixed_variable_treatment=MadNLP.MakeParameter,
)
stats = MadNLP.solve!(solver)
```

## Caveats

- For large-instances, the method might have difficulties to convergence for a tolerance `tol` below `1e-6`.
- The linear solver `cuDSS` is fast, but sometimes lack the robustness required for reliable convergence in IPM. We recommend using the LDL factorization by default.

## References

- The method has been adapted to interior-point by Shaked Regev et al in [the HyKKT paper](hykkt2023).
- [We have extended it recently](nlpgpu2024) and have integrated HyKKT as a MadNLP KKT system. For a broad range of problems, the method is faster than a state-of-the-art linear solver running on the CPU.

If you find this package successful in your work, we would greatly
appreciate you cite this article:
```bibtex
@article{pacaud2024condensed,
  title={Condensed-space methods for nonlinear programming on {GPU}s},
  author={Pacaud, Fran{\c{c}}ois and Shin, Sungho and Montoison, Alexis and Schanen, Michel and Anitescu, Mihai},
  journal={arXiv preprint arXiv:2405.14236},
  year={2024}
}
```

[cops]: https://www.mcs.anl.gov/~more/cops/
[cudss]: https://docs.nvidia.com/cuda/cudss/
[cudssjl]: https://github.com/exanauts/CUDSS.jl
[examodels]: https://github.com/exanauts/ExaModels.jl
[golub2003]: https://epubs.siam.org/doi/abs/10.1137/S1064827500375096
[hykkt2023]: https://www.tandfonline.com/doi/abs/10.1080/10556788.2022.2124990
[madnlp]: https://github.com/MadNLP/MadNLP.jl
[nlpgpu2024]: https://arxiv.org/abs/2405.14236
