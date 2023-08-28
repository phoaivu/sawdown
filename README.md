`sawdown` is an optimization framework, designed for constrained optimization and 
integer programs. Current features include:
1. Unconstrained optimization
2. Linearly-constrained optimization, both equality and inequality constraints.
3. Mixed-Integer Program solver via branch-and-bound.

## API


## Configuration

## Experimental: Symbolic derivatives

`sawdown` comes with built-in symbolic derivation. You define a function containing
the formulation of the objective, and use `.objective_symbolic()` to use it, like so:

```python
import numpy as np

import sawdown as sd
from sawdown import ops

a = np.arange(15).reshape((3, 5))
b = np.ones((3,), dtype=float)
        
def _objective(x, y):
    x_cost = ops.matmul(a, x) + b[:, None]
    x_cost = ops.sum(ops.square(x_cost))
    y_cost = 0.2 * ops.sum(ops.square(y))
    return x_cost + y_cost

optimizer = sd.FirstOrderOptimizer().objective_symbolic(_objective, var_dims=(5, 4)) \
    .fixed_initializer(np.ones((9, ), dtype=float)) \
    .adam() \
    .stop_after(100).stop_small_steps().stream_diary('stdout')

solution = optimizer.optimize()

# The solution (`solution.x`) will be a 9-dimensional vector, specified by `var_dims=(5,4)` and
# the way `_objective(x, y)` takes `x` and `y` as its arguments.
```

In the current state, this feature is limited, with very few operations and less-than-ideal
performance. You can of course use other symbolic derivative libraries and use `.objective_instance()`
to do this in a more efficient way.

The idea behind doing symbolic derivative in `sawdown` is to support optimizers that utilizes 2nd-order
derivatives, but once that happens (although not clear when), it is likely going to be done in a different way.


## Remarks

### Precision
Most numerical optimization algorithms depends on floating-point operations,
which has certain level of precision. In `sawdown`, you can control the precision via
the `.config()` function:

```python
import sawdown

r = sawdown.FirstOrderOptimizer().config(epsilon=1e-24)
```

In general, large `epsilon` allows the algorithm stops sooner, but the solution is
less exact. With `epsilon = 1e-24`, the solution, if any, is roughly precise up to
`1e-12`. The rationale can be seen in `sawdown.stoppers`.

This is different from actual machine precision, which, in Python, is configured via `numpy`.
In general, if `numpy`'s precision is about `1e-48` (which is not the smallest it can take),
the smallest value you can use for `sawdown` is `1e-24`.

For the technical audience, `.stop_small_steps()` would stop optimizing when the relative magnitude
of the step w.r.t. the variable is smaller than a power of `epsilon`. Now if `epsilon`, raised to that 
power, is smaller than `numpy`'s opinionated precision, it will be simply consider zero, 
and the algorithm is not likely to stop.

It is a stretch though. For most practical problems, you would be good with as small as 
`epsilon=1e-14` in `sawdown`.

### Initialization

For linear constraints, `sawdown` can initialize automatically. If 
your problem has at least a linear constraint, you don't
need to specify an initializer.

## Why `sawdown`
This is a photo of `sâu đo` (/ʂəw ɗɔ/), the animal, in Vietnamese. It moves similarly to how most
numerical optimization algorithms work: one step at a time, with the shortest most-effective step.

![image](https://img5.thuthuatphanmem.vn/uploads/2021/11/26/anh-con-sau-do-dep_035620548.jpg)

With a little play of words, it becomes `sawdown`. I later realized
it should better be called `showdoor`, but changes are costly.