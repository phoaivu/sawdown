1. Customizable branch for MIP.
2. Variable scaling:
   1. When constructing the optimizer, select the variables which are bounded 2 ways: `x \in [a, b]`
   2. Create a new Scaler object, stored in the Optimizer
   3. Create the corresponding bound constraints with the canonical bounds: either (0, 1) or (-1, 1)
   4. Optimization loop:
      1. Descale variables: obj, grad = objective_and_gradient(descale(x_k))
      2. Canonicalize the gradient: grad = scale(grad)
      3. Compute the direction and steplength as usual, since the constraints now use the canonical bounds.
      4. Update: x_k += delta * descale(d_k)
      5. Stop conditions are evaluated with scaled x_k and scaled d_k
   5. Changes to the B&B optimizer: None, since the first-order optimizer returns the descaled x_k.
   6. Potential problems:
      1. Unbounded variables, or one-bounded variables would still have gradient with different magnitude.
      2. Linear constraints do not know anything about implicit canonicalization.
      3. Complicate initialization.
      4. When providing APIs to allow users to customize the scaling process (i.e. explicit variable scaling),
      may lead to confusion in writing the objective function, specifying bounds, etc...
   7. Pros:
      1. Less sensitive to linesearch and skewed variable domain.
      2. Useful for problems with many binary and bounded integer variables.

3. Nice-to-have:
   1. Handle KeyboardInterrupt in multiprocessing Workers, for REPL.
   2. Handle ValueError in multiprocessing Workers.

Idea: for integer constraints that limit the feasible region 
into an n-dimensional unitary cube (could be a subset of variables),
may make it into a unitary sphere -> quadratic constraints.