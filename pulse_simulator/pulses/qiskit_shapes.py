import functools
import numpy as np
from qiskit import pulse
from scipy.integrate import simpson

# NOTE: recommed returning qiskit.pulse.Waveform objects
# TODO: Flat top gaussian pulse


def get_amplitude_for_rotation(theta, signal, dt):
    """Get the amplitude of a pulse to achieve a rotation.

    Arguments:
        theta -- Angle to rotate
        signal -- Pulse signal
        dt -- Timestep of the backend

    Returns:
        Amplitude of the pulse.
    """
    return theta / simpson(signal, dx=dt)


def set_limit_amplitude(signal, limit_amplitude):
    # A bug in Qiskit requires this to be set manually for Waveforms
    signal.limit_amplitude = limit_amplitude
    return signal


def gaussian(x, mu=0, sigma=1):
    return np.exp(-((x - mu) ** 2) / 2 / sigma**2)


def lifted_gaussian(x, mu, sigma, x0=-1):
    g = functools.partial(gaussian, mu=mu, sigma=sigma)
    return (g(x) - g(x0)) / (1 - g(x0))


def truncated_gaussian(x, mu=0, sigma=1):
    g = functools.partial(gaussian, mu=mu, sigma=sigma)
    return g(x) - g(0)


def gaussian_envelope(duration, dt=1.0, angle=np.pi, sigma=None, limit_amplitude=False):
    """Define gaussian envelope function to accumulate the angle.

    Arguments:
        duration -- Integer pulse length in terms of the the sampling period dt.

    Keyword Arguments:
        dt -- Timestep of the backend (default: {1.0})
        angle -- Angle to rotate (default: {np.pi})
        limit_amplitude -- Limit the amplitude to 1 (default: {False})

    Returns:
        Qiskit Gaussian pulse implementing angle.
    """
    # Arbitrary default width
    sigma = duration / 4 if sigma is None else sigma

    # Fixed center
    mu = duration / 2

    # Normalize
    signal = lifted_gaussian(np.arange(0, duration), mu, sigma)
    amplitude = get_amplitude_for_rotation(angle, signal, dt)

    # Adjust the amplitude to achieve the angle
    w = pulse.Waveform(signal * amplitude, limit_amplitude=limit_amplitude)
    return set_limit_amplitude(w, limit_amplitude)


def constant_envelope(duration, dt=1.0, angle=np.pi, limit_amplitude=False):
    """Define constant envelope function to accumulate the angle.

    Arguments:
        duration -- Integer pulse length in terms of the the sampling period dt.

    Keyword Arguments:
        dt -- Timestep of the backend (default: {1.0})
        angle -- Angle to rotate (default: {np.pi})
        limit_amplitude -- Limit the amplitude to 1 (default: {False})

    Returns:
        Qiskit constant pulse implementing angle.
    """
    signal = np.ones(duration)
    amplitude = get_amplitude_for_rotation(angle, signal, dt)

    # Adjust the amplitude to achieve the angle
    w = pulse.Constant(duration, amplitude, limit_amplitude=limit_amplitude)
    return set_limit_amplitude(w, limit_amplitude)


def zero_envelope(duration):
    """Define a zero envelope function.

    Arguments:
        duration -- Integer pulse length in terms of the the sampling period dt.

    Returns:
        Qiskit zero pulse.
    """
    return pulse.Constant(duration, 0.0)


def custom_envelope(duration, signal, dt=1.0, angle=None, limit_amplitude=False):
    """Define a custom envelope function.

    Arguments:
        duration -- Integer pulse length in terms of the the sampling period dt.
        signal -- Array of length duration

    Keyword Arguments:
        dt -- Timestep of the backend (default: {1.0})
        angle -- Angle to rotate. If None, use the provided signal.  (default: {None})
        limit_amplitude -- Limit the amplitude to 1 (default: {False})

    Returns:
        Qiskit custom pulse waveform implementing angle.
    """
    assert len(signal) == duration, "Signal must be the same length as duration"

    if angle is None:
        amplitude = 1.0
    else:
        amplitude = get_amplitude_for_rotation(angle, signal, dt)

    w = pulse.Waveform(signal * amplitude, limit_amplitude=limit_amplitude)
    return set_limit_amplitude(w, limit_amplitude)


def resample(duration, signal):
    """Resample a signal to a new duration.

    Arguments:
        duration -- Integer pulse length in terms of the the sampling period dt.
        signal -- Array of length duration

    Returns:
        Resampled signal.
    """
    return np.interp(
        np.linspace(0, len(signal), duration, endpoint=False),
        np.arange(len(signal)),
        signal,
    )
