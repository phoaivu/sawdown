`sawdown` is an optimization framework, designed for constrained optimization and 
integer programs. Current features include:
1. Unconstrained optimization
2. Linearly-constrained optimization, both equality and inequality constraints.
3. Mixed-Integer Program solver via branch-and-bound.

## API


## Configuration

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

### Initialization

For linear constraints, `sawdown` can initialize automatically. Therefore if 
your problem has at least a linear (or fixed-value) constraint, you don't
need to specify an initializer.

## Why `sawdown`
This is a photo of `sâu đo` (/ʂəw ɗɔ/), the animal, in Vietnamese. It moves similarly to how most
numerical optimization algorithms work: one step at a time, with the shortest most-effective step.

![image](https://img5.thuthuatphanmem.vn/uploads/2021/11/26/anh-con-sau-do-dep_035620548.jpg)

With a little play of words, it becomes `sawdown`. I later realized
it should better be called `showdoor`, but changes are costly.