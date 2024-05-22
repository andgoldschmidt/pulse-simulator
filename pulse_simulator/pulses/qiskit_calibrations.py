import numpy as np
from qiskit import pulse
from typing import Union

from .qiskit_shapes import (
    gaussian_envelope,
    get_amplitude_for_rotation,
    get_samples,
    resample,
    scalable_symbolic_pulse,
    zero_envelope,
)


def get_closest_multiple_of(value, base_number):
    return int(value + base_number / 2) - (int(value + base_number / 2) % base_number)


def get_closest_multiple_of_16(num, granularity=16):
    return get_closest_multiple_of(num, granularity)


def qiskit_closest_multiple_of_16(gate_time_si_units, backend):
    # https://qiskit.org/textbook/ch-quantum-hardware/calibrating-qubits-pulse.html
    with pulse.build(name="seconds_to_samples", backend=backend) as _:
        samples = get_closest_multiple_of_16(
            pulse.seconds_to_samples(gate_time_si_units)
        )
    return samples


def calibrate_gateset(
    gateset: list[np.ndarray],
    names: list[str],
    angles: list[float],
    duration: int,
    dt: float,
    samples: Union[int, None] = None,
    verbose: bool = True,
) -> dict:
    """Calibrate a gateset of pulses."""
    assert len(angles) == len(gateset), "Number of angles must match number of gates."
    assert len(names) == len(gateset), "Number of names must match number of gates."

    if verbose:
        print("Storing symbolic pulses...")

    # Note: If too long, lower samples
    # Create symbolic pulses
    stored_envs = []
    for i, v in enumerate(gateset):
        env = resample(
            len(v) if samples is None else samples,
            v,
            keep_step=False,
        )
        stored_envs.append(
            scalable_symbolic_pulse(
                duration, env, f"memoize-{i}", limit_amplitude=False
            )
        )

    calibrated_gateset = {}
    for i, v in enumerate(gateset):
        # Get calibrated amplitude
        try:
            # NOTE: Real-valued signals only
            samples = get_samples(stored_envs[i]).real
            amp = get_amplitude_for_rotation(angles[i], samples, dt)
            message = f"Adjusting {names[i]} by amplitude{amp: 0.2f} for angle{angles[i]/np.pi: 0.2f}π"
        except ValueError:
            # Catch failed amplitude calculation
            amp = angles[i]
            message = f"Zero area signal. Adjusting {names[i]} by amplitude{amp: 0.2f} (using provided angle)."

        if verbose:
            print(message)

        # Apply to gate
        calibrated_gateset[names[i]] = pulse.ScalableSymbolicPulse(
            pulse_type=names[i],
            duration=duration,
            amp=amp,
            angle=0.0,
            limit_amplitude=False,
            envelope=stored_envs[i].envelope,
        )

    return calibrated_gateset


def default_single_qubit_gateset(
    keys: list[str],
    duration: int,
    dt: float,
    sx_angle: float = np.pi / 4,
    x_angle: float = np.pi / 2,
):
    gates = {}
    for k in keys:
        if "SX" in k:
            gates[k] = gaussian_envelope(duration, dt=dt, angle=sx_angle)
        elif "X" in k:
            gates[k] = gaussian_envelope(duration, dt=dt, angle=x_angle)
        elif "I" in k:
            gates[k] = zero_envelope(duration)
        else:
            raise ValueError(f"Unknown gate {k}")
    return gates
