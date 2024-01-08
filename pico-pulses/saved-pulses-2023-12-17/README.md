# Two qubit pulses

The pulses should be crosstalk robust. They are just dephasing robust for two qubit gate lengths. The target is a 93.3 ns $R_{zx}$ gate with angle $\frac{π}{4}$ (two such rotatings in Echo CR provide the desired $\frac{π}{2}$).

In the pulses in this folder, adjust units by factor of 10 (ns * 10, GHz / 10). Resample if necessary for use in Qiskit.

| Variable | Value |
|----------|-------|
| T        | 84    |
| Δt       | 1/9   |


|          | Default | Robust   |
|----------|---------|----------|
| Max control | 0.09    | 0.46  |
| Robustness  | 0.18    | 0.0017   |
