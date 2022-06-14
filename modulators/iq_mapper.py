import numpy as np
from scipy.spatial.distance import cdist


# iq mapping implementation from the book:
# Digital Modulations using Python by Mathuranathan Viswanathan
def iq_mapping(received_symbols, constellation):
    """
    Optimum Detector for 2-dim. signals (ex: MQAM,MPSK,MPAM) in IQ Plane
    Note: MPAM/BPSK are one dimensional modulations. The same function can be
    applied for these modulations since quadrature is zero (Q=0)
    The function computes the pair-wise Euclidean distance of each point in the
    received vector against every point in the reference constellation. It then
    returns the symbols from the reference constellation that provide the
    minimum Euclidean distance.
    Parameters:
    receivedSyms : received symbol vector of complex form
    Returns:
    detectedSyms : decoded symbols that provide minimum Euclidean distance
    """

    # received vector and reference in cartesian form
    XA = np.column_stack((np.real(received_symbols), np.imag(received_symbols)))
    XB = np.column_stack((np.real(constellation), np.imag(constellation)))
    d = cdist(XA, XB, metric='euclidean')  # compute pair-wise Euclidean distances
    detected_symbols = np.argmin(d, axis=1)  # indices corresponding minimum Euclid. dist.
    return detected_symbols
