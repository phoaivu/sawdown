1. Customizable branch for MIP.
2. Separated worker for writing logs.
3. Bound constraints, to reduce computational overhead.

Idea: for integer constraints that limit the feasible region 
into an n-dimensional unitary cube (could be a subset of variables),
may make it into a unitary sphere -> quadratic constraints.