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


def seconds_to_samples(gate_time, dt):
    return int(gate_time / dt)


def get_closest_multiple_of(value, base_number):
    return int(value + base_number / 2) - (int(value + base_number / 2) % base_number)


def get_closest_multiple_of_16(num, granularity=16):
    return get_closest_multiple_of(num, granularity)


def qiskit_closest_multiple_of_16(desired_samples_dt):
    # https://qiskit.org/textbook/ch-quantum-hardware/calibrating-qubits-pulse.html
    return get_closest_multiple_of_16(desired_samples_dt)


def calibrate_gateset(
    gateset: list[np.ndarray],
    names: list[str],
    angles: list[float],
    duration: int,
    dt: float,
    samples: Union[int, None] = None,
    verbose: bool = True,
) -> dict:
    """Calibrate a gateset of pulses.
    
    Args:

    gateset (list[np.ndarray]): List of gates to calibrate.
    names (list[str]): List of gate names.
    angles (list[float]): List of rotation angles.
    duration (int): Duration of the pulses.
    dt (float): Time step.
    samples (int, optional): Number of samples to store in the spline fit. 
        Defaults to None, which uses the number of samples in the provided gateset.
    verbose (bool, optional): Print messages. Defaults to True.

    Returns:
    dict: Calibrated gateset.


    
    """
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
            message = (
                f"Adjusting {names[i]} by amplitude {amp: 0.2f}"
                + f" for rotation angle {angles[i]/np.pi: 0.2f}π"
            )
        except ValueError:
            # Catch failed amplitude calculation
            amp = angles[i]
            message = (
                f"Zero area signal. Adjusting {names[i]} by amplitude {amp: 0.2f}"
                + " (using provided rotation angle)."
            )

        if verbose:
            print(message)

        # Apply to gate
        if verbose:
            if type(duration) != int:
                print(f"Duration is not an integer: {duration}, converting.")
                duration = int(duration)

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
):
    gates = {}
    for k in keys:
        if "SX" in k:
            gates[k] = np.pi / 4
        elif "X" in k:
            gates[k] = np.pi / 2
        elif "I" in k:
            gates[k] = 0.0
        else:
            raise ValueError(f"Unknown gate {k}")
    return default_gateset(gates, duration, dt)


def default_gateset(
    gates: dict[str, float],
    duration: int,
    dt: float,
):
    gateset = {}
    for k, v in gates.items():
        if np.isclose(v, 0.0):
            gateset[k] = zero_envelope(duration)
        else:
            gateset[k] = gaussian_envelope(duration, dt=dt, angle=v)
    return gateset
