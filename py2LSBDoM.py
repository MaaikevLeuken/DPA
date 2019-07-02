"""
DPA attack on AES.
The difference of means is used as distinguisher,
with the least significant bit as power model.

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

Step 4 of the DPA attack, using the least significant bit as power model.
Result: D by K matrix H.
"""


# Calculate the hypothetical power consumption values, a D x K matrix H.
# For all plaintexts and key possibilities, compute LSB(Sbox(plaintext XOR key)).
def LSB_hypo_power():
    h = [[int("{0:08b}".format(int(v[d][k], 16))[7]) for k in range(K)] for d in range(D)]
    return h


# Computes the ith column of a given array.
def column(array, i):
    return [row[i] for row in array]


# Subtracts 1 by the ith column of a given array.
def neg_column(array, i):
    return [1 - row[i] for row in array]


"""
RESULTS

Step 5 of the DPA attack.
Result: K by T matrix R.
"""


# Compute the K x T result matrix R using the difference of means.
def dom():
    R = []
    T0 = []
    T1 = []
    for k in range(K):
        for d in range(D):
            if h[d][k] == 0:
                T0.append(traces[d])
            else:
                T1.append(traces[d])
        M1 = np.einsum("d,dt->t", column(h, k), traces, optimize='optimal') / np.double(len(T1))
        M0 = np.einsum("d,dt->t", neg_column(h, k), traces, optimize='optimal') / np.double(len(T0))
        R.append(M1 - M0)
        T1 = []
        T0 = []
    return R


# Find the highest correlation in R. The indices correspond to the correct key
# and the time at which the SubBytes round was processed.
def max_absolute(result):
    highest_cor = 0
    a = -1
    b = -1
    for row in range(K):
        x = max(map(abs, result[row]))
        if x > highest_cor:
            highest_cor = x
            a = row
    for column in range(T_used):
        if abs(result[a][column]) == highest_cor:
            b = column
    return a, b, highest_cor


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
    for byte in range(16):
        v = hypo_inter_values(byte)
        print "Hypothetical intermediate values computed."
        h = LSB_hypo_power()
        print "Hypothetical power consumptions computed."
        result = dom()
        print "Distance of Means computed."
        (i, j, highest_cor) = max_absolute(result)
        print "Byte " + str(byte) + " has been processed."
        key.append((keys[i], j, highest_cor))
    main()
