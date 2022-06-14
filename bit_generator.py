import numpy as np


def bit_generator(num_bits):
    """Generates an array of random bits with size 'num_bits'."""
    bits = np.random.randint(2, size=num_bits)    # array of data bits to be sent
    return bits


def corrupt(bits, ber):
    """Corrupts bits in the given array with a given bit error rate.
       Used for testing."""
    rng = np.random.default_rng()
    corrupted_bits = np.zeros(len(bits))
    for i, bit in enumerate(bits):
        corrupted_bits[i] = bit
        if rng.random() <= ber:
            if bit == 1:
                corrupted_bits[i] = 0
            else:
                corrupted_bits[i] = 1
    return corrupted_bits
