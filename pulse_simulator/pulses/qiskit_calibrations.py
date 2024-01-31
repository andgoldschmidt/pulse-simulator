from qiskit import pulse


def get_closest_multiple_of(value, base_number):
    return int(value + base_number / 2) - (int(value + base_number / 2) % base_number)


def get_closest_multiple_of_16(num, granularity=16):
    return get_closest_multiple_of(num, granularity)


def qiskit_closest_multiple_of_16(gate_time_si_units, backend):
    with pulse.build(name="seconds_to_samples", backend=backend) as _:
        samples = get_closest_multiple_of_16(
            pulse.seconds_to_samples(gate_time_si_units)
        )
    return samples
