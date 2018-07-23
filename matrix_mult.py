"""
A and B are arbitrary matrices that may be multiplied
"""

def multiply(A, B):
    out_dim_1 = len(A)
    out_dim_2 = len(B[0])

    H = [[0 for a in range(out_dim_2)] for b in range(out_dim_1)]
    for i in out_dim_1:
        for j in out_dim_2:
            for k in len(B):
                H[i][j] += A[i][k] * B[k][j]
    return H