# Robust two qubit gate

https://research.ibm.com/publications/procedure-for-systematically-tuning-up-cross-talk-in-the-cross-resonance-gate

"We were able to benchmark results showing a CR gate fidelity 0.991 ± 0.002 for a 160 ns gate. This gate time includes 20 ns added by the single qubit echo, which is buffered by two 10 ns delays. The CR pulses are rounded square pulses with Gaussian rise times of 15 ns."

From this, we conclude the CR gate is 160 - 20 - 10 - 10 = 120.

Each ZX rotation is half this, so the pulses should implement a 60ns ns π/4 gate (angle is π/8, i.e. e^{-i π/8 ZX}).

Here are the stats:
T = 270
Δt = 2/9
T * Δt = 60
