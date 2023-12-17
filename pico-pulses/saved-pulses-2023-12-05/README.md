# Single qubit gateset

__Number of timesteps__ 50

__Timestep__ 0.2 ns

__Duration__ 10 ns

For the following table, R is the regularization placed on the control and the acceleration.
The default has no robustness optimization.

The regularized controls are initialized after optimizing the default getset using an objective that includes fidelity and crosstalk robust control. The original problem is initialized with π/2/10 to capture a single rotation over the duration 10. The default gateset is modified by prefactors (1 + π, 1) for each pair of gates according to the theory of crosstalk robust pairs of gates[^1].

| __R__          | Default  | 1e-1     | 1e-2     | 1e-3     |  1e-5    | 1e-6     |
| -------------- | -------- | -------- | -------- | -------- | -------- | -------- |
| __Crosstalk__  | 3.075e-2 | 9.257e-3 | 4.733e-3 | 6.152e-4 | 4.159e-5 | 1.788e-5 |
| __Fidelity__   | 0.9999   | 0.9999   | 0.99899  | 0.9999   | 0.9999   | 0.9999   |
| __max(a)__     | 0.2074   | 0.59537  | 0.60632  | 0.8783   | 1.1297   | 1.2240   |
| __max(dda)__   | 0.0448   | 0.02479  | 0.29606  | 0.6603   | 1.7607   | 3.2215   |


We could modify this optimization problem to enforce bounds on the max of the control and acceleration. However, this leads to more challenging optimizations, in the sense that even when the limits are set to ±5.0, robustness is harder to minimize to values like 1e-5.

Another missing experiment here is to rescale the default controls so that we use flat-top Gaussian controls that are offset in the way that makes the robustness factor zero.

[^1] Check this, it doesn't matter for optimization, but might be slightly off what I want.