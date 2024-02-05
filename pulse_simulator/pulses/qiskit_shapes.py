import functools
import numpy as np
from qiskit import pulse
from scipy.integrate import simpson
import sympy

# TODO: Flat top gaussian envelope
# TODO: Refactor angle to area (because qiskit uses kw angle for phase)
# TODO: Bug in SymbolicPulse.get_waveform(): Limit amplitude not passed.


def get_samples(obj):
    if isinstance(obj, pulse.Waveform):
        return obj.samples
    elif isinstance(obj, pulse.SymbolicPulse):
        return obj.get_waveform().samples
    elif isinstance(obj, np.ndarray):
        return obj
    else:
        raise TypeError("Object is not a pulse waveform or symbolic pulse.")


def get_amplitude_for_rotation(theta, signal, dt):
    """Get the amplitude of a pulse to achieve a rotation.

    Arguments:
        theta -- Angle to rotate
        signal -- Pulse signal
        dt -- Timestep of the backend

    Returns:
        Amplitude of the pulse.
    """
    if len(signal) == 0:
        raise ValueError("Signal is empty.")
    elif len(signal) == 1:
        return theta
    else:
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
        duration -- Integer pulse length in terms of the sampling period dt.

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


def constant_envelope(duration, dt=1.0, angle=np.pi, name=None, limit_amplitude=False):
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
    s = pulse.Constant(duration, amplitude, name=name, limit_amplitude=limit_amplitude)
    try:
        return s.get_waveform()
    except pulse.PulseError:
        # Older versions of Qiskit do not pass limit_amplitude correctly.
        return pulse.Waveform(
            amplitude * np.ones(duration), name=name, limit_amplitude=limit_amplitude
        )


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
        duration -- Desired pulse length in terms of the sampling period dt.
        signal -- Array of samples. If the length does not match the duration,
            the signal is resampled.

    Keyword Arguments:
        dt -- Timestep of the backend (default: {1.0}). Only used if angle is set.
        angle -- Angle to rotate. If None, use the provided signal.  (default: {None})
        limit_amplitude -- Limit the amplitude to 1 (default: {False})

    Returns:
        Qiskit custom pulse waveform implementing angle.
    """
    # Resample the signal to the new duration
    if len(signal) != duration:
        signal = resample(duration, signal)

    # If angle is not provided, use the signal as is
    if angle is None:
        amplitude = 1.0
    else:
        amplitude = get_amplitude_for_rotation(angle, signal, dt)

    w = pulse.Waveform(amplitude * signal, limit_amplitude=limit_amplitude)
    return set_limit_amplitude(w, limit_amplitude)


def resample(duration, signal, keep_step=True, return_step=False):
    """Resample a signal to a new duration. Default behavior is to keep the
    timestep and integral of the original signal, so integating the returned
    signal with the original timestep will return the original area. Equivalently,
    when keep_step is True, the amplitude of the resampled signal is scaled by
    the ratio of the new and old timesteps.

    Arguments:
        duration -- Integer pulse length in terms of the sampling period dt.
        signal -- Array of length duration

    Keyword Arguments:
        return_step -- Return the timestep of the resampled signal (default: {False})
        keep_step -- Preserve the timestep of the original signal (default: {True})

    Returns:
        Resampled signal.
    """
    new_times = np.linspace(0, len(signal), duration, endpoint=False)
    Δt = new_times[1] - new_times[0]
    resample = np.interp(
        new_times,
        np.arange(len(signal)),
        signal,
    )
    # Rescale the area to be the same as the original signal
    resample = resample * Δt if keep_step else resample
    if return_step:
        return resample, Δt
    else:
        return resample


@functools.cache
def fit_spline_envelope(t_tuple, y_tuple, degree):
    """Fit a spline to the data. Tuples are immutable. Cache the spline fit to
    avoid recomputation.

    Arguments:
        t_tuple -- Tuple of times
        y_tuple -- Tuple of amplitudes
        degree -- Degree of Bspline strictly greater than equal to one

    Returns:

    """
    # Required symbols for Qiskit SymbolicPulse
    _t, _amp, _angle = sympy.symbols("t, amp, angle")

    t_list = list(t_tuple)
    y_list = list(y_tuple)
    spline_env = sympy.interpolating_spline(degree, _t, t_list, y_list)
    env = (
        _amp
        * sympy.exp(sympy.I * _angle)
        * sympy.Piecewise((spline_env, sympy.And(0 <= _t, _t <= t_list[-1])), (0, True))
    )
    return env


def scalable_symbolic_pulse(
    duration: int,
    signal: np.ndarray | pulse.Waveform | pulse.SymbolicPulse,
    pulse_type: str,
    amp=1,
    angle=0,
    degree=3,
    name=None,
    limit_amplitude=False,
) -> pulse.ScalableSymbolicPulse:
    """
    Use spline interpolation to create a symbolic pulse from the provided signal.

    Fit is given by spline of degree, passing through the given signal over the
    provided duration. The spline fit is memoized to avoid recomputation. The
    spline is defined to be zero outside of the provided duration.

    The provided signal can be any length. If the length does not match the
    duration, the timesteps are rescaled.

    Scalable symbolic pulses are useful when pulse envelopes are able to
    be scaled to different amplitudes and angles. The pulse envelope is
    defined using the provided signal, which is cached. Later, amplitude and
    angle can be used to quickly rescale this envelope.

    TODO: steps where=['post', 'pre', 'mid']

    Arguments:
        duration -- Integer pulse length in terms of the sampling period dt.
        signal -- Array, Waveform, or SymbolicPulse representing the envelope.
        pulse_type -- Display name of this pulse shape.

    Keyword Arguments:
        name --  Display name for this particular pulse envelope. (default: {None})
        degree -- Degree of Bspline strictly greater than equal to one (default: {3})

    Returns:
        The ScalableSymbolicPulse representation model of the signal.

    """
    assert isinstance(
        signal, (np.ndarray, pulse.Waveform, pulse.SymbolicPulse)
    ), "Invalid signal type."
    samples = get_samples(signal)

    if duration != len(samples):
        # Timesteps are *not* in terms of the sampling period dt.
        Δt = duration / len(samples)
    else:
        Δt = 1.0

    if isinstance(signal, pulse.SymbolicPulse):
        env = signal.envelope
    else:
        # Bump step by one for interpolation intervals
        # 'post' steps
        y_list = samples.tolist()
        ys = tuple(y_list + [0.0])
        ts = tuple(Δt * np.arange(0, len(samples) + 1))

        # fit a spline to the data (extrapolate with zeros)
        env = fit_spline_envelope(ts, ys, degree)

    return pulse.ScalableSymbolicPulse(
        pulse_type=pulse_type,
        duration=duration,
        amp=amp,
        angle=angle,
        parameters=None,
        name=name,
        limit_amplitude=limit_amplitude,
        envelope=env,
        constraints=None,
        valid_amp_conditions=None,
    )
