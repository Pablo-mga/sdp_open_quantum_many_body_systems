import cvxpy as cp
import numpy as np
from scipy.linalg import expm
from scipy.sparse import kron, identity, csr_matrix
from scipy.sparse.linalg import eigs,eigsh,expm_multiply
import matplotlib.pyplot as plt
import pandas as pd
import os
import time


# Define Pauli matrices
sx = csr_matrix(np.array([[0, 1], [1, 0]], dtype=complex))
sy = csr_matrix(np.array([[0, -1j], [1j, 0]], dtype=complex))
sz = csr_matrix(np.array([[1, 0], [0, -1]], dtype=complex))
id2 = identity(2, format='csr', dtype=complex)

pauli_matrices=[id2,sx,sy,sz]


def kron_n(ops):
    """Kronecker product of a list of operators."""
    result = ops[0]
    for op in ops[1:]:
        result = kron(result, op, format='csr')
    return result

def op_support(L_op, ini, k, L):
    """
    Build operators acting on qubits
    ini, ..., ini+k-1 with condition ini+k<=L.

    Parameters
    ----------
    L_op : dict
        [{"ops": [csr_matrix,...], "sites": [i,...]}]
    ini : int
        initial qubit
    k : int
        number of qubits to be considered
    L : int
        total number of qubits

    Returns
    -------
    list of csr_matrix that can completely act on the qubits
        operators 2^k x 2^k
    """
    if ini + k > L:
        raise ValueError("ini + k must be less than or equal to L.")
        
    L_op_k = []

    for term in L_op:
        ops_term = term["ops"]
        sites = term["sites"]

        local_sites = [ s - ini for s in sites ]

        if all( 0<= ls < k for ls in local_sites):
            ops = [id2] * k
            for ls, op in zip(local_sites, ops_term):
                ops[ls] = op
                
            
            L_op_k.append(kron_n(ops))

    return L_op_k



    


