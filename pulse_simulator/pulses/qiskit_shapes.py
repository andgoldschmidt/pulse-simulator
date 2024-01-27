import functools
import numpy as np
from qiskit import pulse

# TODO: Flat top gaussian pulse


def gaussian(x, mu=0, sigma=1):
    return np.exp(-((x - mu) ** 2) / 2 / sigma**2)


def lifted_gaussian(x, mu, sigma, x0=-1):
    g = functools.partial(gaussian, mu=mu, sigma=sigma)
    return (g(x) - g(x0)) / (1 - g(x0))


def truncated_gaussian(x, mu=0, sigma=1):
    g = functools.partial(gaussian, mu=mu, sigma=sigma)
    return g(x) - g(0)


def gaussian_envelope(dt, duration, angle=np.pi):
    """Define gaussian envelope function to accumulate the angle.

    Arguments:
        dt -- Timestep of the backend
        duration -- Duration of the gate

    Keyword Arguments:
        angle -- Angle to rotate (default: {np.pi})

    Returns:
        Qiskit Gaussian pulse implementing angle.
    """

    steps = int(duration / dt)

    # Arbitrary shape
    # NOTE: Qiskit doesn't like pulse amplitudes > 1 by default
    sigma = steps / 4
    mu = steps / 2

    # Normalize
    x = np.linspace(0, steps, endpoint=True)
    area = np.trapz(lifted_gaussian(x, mu, sigma), x)
    amplitude = angle / area / dt

    # Adjust the amplitude to achieve the angle
    return pulse.Gaussian(steps, amplitude, sigma, limit_amplitude=False)


def zero_envelope(dt, duration):
    """Define a zero envelope function.

    Arguments:
        dt -- Timestep of the backend
        duration -- Duration of the gate

    Returns:
        Qiskit zero pulse.
    """
    steps = int(duration / dt)
    return pulse.Constant(steps, 0.0)
