import matplotlib.pyplot as plt


def plot_pulse_schedule(
        signals,
        pulse_schedule,
        duration,
        figsize=None):
    """ Plot a grid of pulses showing envelopes and carried signals.

    Arguments:
        signals (List[qiskit_dynamics.signals.signals.DiscreteSignal]) -- Pulse
            signals
        pulse_schedule (qiskit.pulse.schedule.ScheduleBlock) -- Pulse schedule
        duration [Optional] (float) -- Max time for x-axis of plot

    Returns:
        Figure and axes
    """
    fig, axes = plt.subplots(
        len(pulse_schedule.channels),
        2,
        figsize=figsize
    )

    duration = pulse_schedule.duration if duration is None else duration

    for i, ch in enumerate(pulse_schedule.channels):
        for ax, title in zip(axes[i], ["envelope", "signal"]):
            signals[i].draw(
                0,
                duration,
                1000,
                title,
                axis=ax
            )
            ax.set_xlabel("Time (ns)")
            ax.set_ylabel("Amplitude (GHz)")
            ax.set_title(f"{title}: {ch.name}")
    return fig, axes
