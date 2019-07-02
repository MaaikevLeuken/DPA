"""
CPA attack on AES.
The correlation coefficient is used as distinguisher,
with the Hamming weight as power model.

Maaike
"""

from __future__ import division
import itertools
import numpy as np

"""
TRACES

The class trs2npz parses a .trs file into a .npz file. 
We only need to parse the traces set once.
Include import trs2npz.
"""

# trs2npz.main("traces")


# The byte substitution look-up table in hexadecimal notation for the SubBytes round.
Sbox = (
    ['63', '7C', '77', '7B', 'F2', '6B', '6F', 'C5', '30', '01', '67', '2B', 'FE', 'D7', 'AB', '76'],
    ['CA', '82', 'C9', '7D', 'FA', '59', '47', 'F0', 'AD', 'D4', 'A2', 'AF', '9C', 'A4', '72', 'C0'],
    ['B7', 'FD', '93', '26', '36', '3F', 'F7', 'CC', '34', 'A5', 'E5', 'F1', '71', 'D8', '31', '15'],
    ['04', 'C7', '23', 'C3', '18', '96', '05', '9A', '07', '12', '80', 'E2', 'EB', '27', 'B2', '75'],
    ['09', '83', '2C', '1A', '1B', '6E', '5A', 'A0', '52', '3B', 'D6', 'B3', '29', 'E3', '2F', '84'],
    ['53', 'D1', '00', 'ED', '20', 'FC', 'B1', '5B', '6A', 'CB', 'BE', '39', '4A', '4C', '58', 'CF'],
    ['D0', 'EF', 'AA', 'FB', '43', '4D', '33', '85', '45', 'F9', '02', '7F', '50', '3C', '9F', 'A8'],
    ['51', 'A3', '40', '8F', '92', '9D', '38', 'F5', 'BC', 'B6', 'DA', '21', '10', 'FF', 'F3', 'D2'],
    ['CD', '0C', '13', 'EC', '5F', '97', '44', '17', 'C4', 'A7', '7E', '3D', '64', '5D', '19', '73'],
    ['60', '81', '4F', 'DC', '22', '2A', '90', '88', '46', 'EE', 'B8', '14', 'DE', '5E', '0B', 'DB'],
    ['E0', '32', '3A', '0A', '49', '06', '24', '5C', 'C2', 'D3', 'AC', '62', '91', '95', 'E4', '79'],
    ['E7', 'C8', '37', '6D', '8D', 'D5', '4E', 'A9', '6C', '56', 'F4', 'EA', '65', '7A', 'AE', '08'],
    ['BA', '78', '25', '2E', '1C', 'A6', 'B4', 'C6', 'E8', 'DD', '74', '1F', '4B', 'BD', '8B', '8A'],
    ['70', '3E', 'B5', '66', '48', '03', 'F6', '0E', '61', '35', '57', 'B9', '86', 'C1', '1D', '9E'],
    ['E1', 'F8', '98', '11', '69', 'D9', '8E', '94', '9B', '1E', '87', 'E9', 'CE', '55', '28', 'DF'],
    ['8C', 'A1', '89', '0D', 'BF', 'E6', '42', '68', '41', '99', '2D', '0F', 'B0', '54', 'BB', '16'],
)


# Compute all key hypotheses. There are K = 256 possibilities for a key byte.
def init_keys():
    keys = list(itertools.product([0, 1], repeat=8))
    for poss in range(K):
        keys[poss] = hex(int(''.join(map(str, keys[poss])), 2))  # Rewrite to hexadecimal.
    return keys


"""
HYPOTHETICAL INTERMEDIATE VALUES

Step 3 of the DPA attack. 
Result: D by K matrix V.
"""


# Calculate the hypothetical intermediate values, a D x K matrix V.
# For all plaintexts and key possibilities, compute Sbox(plaintext XOR key).
def hypo_inter_values(byte):
    v = []
    for i in range(D):
        a = data[i][byte]
        x = ["0x" + Sbox[(a ^ int(keys[k], 16)) // 16][(a ^ int(keys[k], 16)) % 16] for k in range(K)]
        v.append(x)
    return v


"""
HYPOTHETICAL POWER CONSUMPTION VALUES

Step 4 of the DPA attack, using the Hamming weight model.
Result: D by K matrix H.
"""


# Counts the amount of logical ones in a byte in binary.
def counter(b):
    count = 0
    for i in range(len(b)):
        if b[i] == "1":
            count += 1
    return count


# Calculate the hypothetical power consumption values, a D x K matrix H.
# For all plaintexts and key possibilities, compute HW(Sbox(plaintext XOR key)).
def hw_hypo_power():
    h = []
    for pt in range(D):
        x = [counter("{0:b}".format(int(v[pt][k], 16))) for k in range(K)]
        h.append(x)
    h = np.array(h)
    return h


"""
RESULTS

Step 5 of the DPA attack.
Result: K by T matrix R.
"""


# Compute the K x T result matrix R using the correlation coefficient.
def correlation():
    diff_t = traces - (np.einsum("dt->t", traces, optimize='optimal') / np.double(D))
    diff_h = h - (np.einsum("dk->k", h, optimize='optimal') / np.double(D))
    cov = np.einsum("dk,dt->kt", diff_h, diff_t, optimize='optimal')  # Covariant.
    std_h = np.einsum("dk,dk->k", diff_h, diff_h, optimize='optimal')  # Standard deviation of h.
    std_t = np.einsum("dt,dt->t", diff_t, diff_t, optimize='optimal')  # Standard deviation of t.
    temp = np.einsum("k,t->kt", std_h, std_t, optimize='optimal')
    return cov / np.sqrt(temp)


# Find the highest correlation in R. The indices correspond to the correct key
# and the time at which the SubBytes round was processed.
def max_absolute(result):
    highest_cor = 0
    i = -1
    j = -1
    for row in range(256):
        x = max(map(abs, result[row]))
        if x > highest_cor:
            highest_cor = x
            i = row
    for column in range(T_used):
        if abs(result[i][column]) == highest_cor:
            j = column
    return i, j, highest_cor


def main():
    help = ""
    for byte in range(16):
        help += key[byte][0]
    print help


if __name__ == "__main__":
    D = 10000
    T_used = 20000
    T = 110000
    K = 256
    key = []
    dic = np.load('traces.npz', mmap_mode='r')
    data = dic['data']
    traces = dic['traces'][:, :T_used]
    keys = init_keys()
    for byte in range(16):  # For every byte, find the key, time and correlation.
        v = hypo_inter_values(byte)
        print "Hypothetical intermediate values computed."
        h = hw_hypo_power()
        print "Hypothetical power consumptions computed."
        result = correlation()
        print "Correlation computed."
        (i, j, highest_cor) = max_absolute(result)
        print "Byte " + str(byte) + " has been processed."
        key.append((keys[i], j, highest_cor))
    main()
