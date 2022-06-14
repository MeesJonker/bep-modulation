import sk_dsp_comm.fec_block as block


class Hamming:
    """Class that implements Hamming codes through the sk_dsp_comm module."""
    def __init__(self, j):
        """Parameter j represents the desired number of parity bits."""
        self.h = block.FECHamming(j)
        self.j = j

    def __str__(self):
        return f"Hamming encoder/decoder: k = {self.h.k}, n = {self.h.n}, j = {self.h.j}, code rate = {self.h.k/self.h.n}"

    def encode(self, data):
        codewords = self.h.hamm_encoder(data)
        return codewords

    def decode(self, codewords):
        decoded_data = self.h.hamm_decoder(codewords.astype(int))
        return decoded_data


