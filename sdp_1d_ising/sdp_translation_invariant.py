import cvxpy as cp
import numpy as np
from scipy.sparse import kron, csr_matrix
import pandas as pd
import os
import time

from utils_1d import sx,sy,sz,id2,kron_n,op_support

t0 = time.time()

pauli_matrices=[id2,sx,sy,sz]


def build_hamiltonian(k, J, g,ini=0):
    H = csr_matrix((2**k, 2**k), dtype=complex)

    # Build J * sum sigma^z_i sigma^z_{i+1}
    for i in range(k-1):
        ops = [id2] * k
        ops[i] = sz
        ops[(i + 1)] = sz
        H += J * kron_n(ops)
    
    # Build g * sum sigma^x_i
    for i in range(k):
        ops = [id2] * k
        ops[i] = sx
        H += g * kron_n(ops)
    
    return H


#adjoint lindbladian as superoperator
def adj_lindblad(X, H, L_op,k):
    """
    Parameters
    ----------
    X,H : matrices 2**k*2**k
    L_op : list of matrices 2**k*2**k
    k : int
        number of qubits to be considered
    """
    L_op_d=[]
    
    for i,L_i in enumerate(L_op):
        L_op_d.append(L_op[i].getH()) 
    
    # Hamiltonian part
    comm = 1j * (H @ X - X @ H)
    
    # Dissipation
    dissipator = csr_matrix((2**k, 2**k), dtype=complex)

    for i in range(len(L_op)):
        dissipator+= L_op_d[i] @ X @ L_op[i] -0.5*(L_op_d[i] @ L_op[i] @ X + X @ L_op_d[i] @ L_op[i])
    
    return comm + dissipator



def translation_invariant_sdp(k,L_op,local_obs,):
    
    #define sdp viariables
    rho = cp.Variable((2**k, 2**k), hermitian=True)  #the quantum state
    #define constriants for the quantum state
    constraints = []
    constraints += [rho >> 0]
    constraints += [cp.trace(rho) == 1]
    constraints += [cp.partial_trace(rho, [2,]*k, 0) ==  \
                    cp.partial_trace(rho, [2,]*k, k-1)]
    
    
    # basis of operators in k-2 internal qubits
    if(k>2):
        X_basis = pauli_matrices.copy()
    
        for _ in range(1, k-2):
            new_basis = []
            for X in X_basis:
                for p in pauli_matrices:
                    new_basis.append(kron(X, p, format='csr'))
            X_basis = new_basis
        
        for i in range(len(X_basis)):
            X_basis[i] = kron_n([id2,X_basis[i],id2])
    elif(k==2):
        X_basis=[]
        X_basis.append(kron_n([id2,id2]))
        
    #compute lindbladian constraints
    H=build_hamiltonian(k, J, g)
    L_op_k = op_support(L_op, 0, k, L)
    for X in X_basis:
        constraints += [cp.trace(adj_lindblad(X, H, L_op_k,k)@rho) == 0]
        
    #express observable as a matrix
    obs=op_support([local_obs], 0, k, L)[0]
    
    #finding the minimum
    objective=cp.Minimize(cp.real(cp.trace(obs@rho)))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.MOSEK, verbose=False,canon_backend=cp.SCIPY_CANON_BACKEND)#set verbose to true to see details on the calculation

    min_obs=prob.value
    print('k= ',k)
    print("problem status = ",prob.status)
    print("Min of obs: ", min_obs)
    print("Time: ", time.time()-t0,"\n")
    
    
    #finding the maximum
    objective=cp.Maximize(cp.real(cp.trace(obs@rho)))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.MOSEK, verbose=False,canon_backend=cp.SCIPY_CANON_BACKEND)#set verbose to true to see details on the calculation

    max_obs=prob.value
    print("problem status = ",prob.status)
    print("Max of obs:", max_obs)
    print("Time: ", time.time()-t0,"\n")
    
    return min_obs,max_obs


def compute_and_save_sdp_limits(local_obs, obs_label, filename, k):

    min_obs, max_obs = translation_invariant_sdp(k, L_op, local_obs)

    data = {
        "k": k,
        "min": min_obs,
        "max": max_obs,
        "obs": obs_label,
        "pos": local_obs["sites"][0],
    }

    pd.DataFrame([data]).to_csv(
        filename,
        mode="a",
        header=not os.path.exists(filename),
        index=False,
    )



'''
Example of a system
'''

# Parameters of the system
L = 10
J = 0.5
g = 0.5
gamma = 1

# Lindbladian operators
L_op = []
for i in range(L):
    op = np.sqrt(gamma) * 0.5 * (sx - 1j * sy)
    L_op.append({
        "ops": [op],
        "sites": [i],
    })

# Directory for results
results_dir = "results_translation_invariant"
os.makedirs(results_dir, exist_ok=True)

# Local observables
observables = {
    "sx": sx,
    "sy": sy,
    "sz": sz,
}

k = 4

for obs_label, obs in observables.items():

    filename = os.path.join(results_dir, f"results_{obs_label}.csv")
    
    # Uncomment to overwrite previous results
    # if os.path.exists(filename):
    #     os.remove(filename)

    local_obs = {
        "ops": [obs],
        "sites": [0],
    }

    compute_and_save_sdp_limits(
        local_obs,
        obs_label,
        filename,
        k,
    )
        
    


