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
    

    # Build h * sum sigma^x_i
    for i in range(k):
        ops = [id2] * k
        ops[i] = sx
        H += g* kron_n(ops)

    #return the Hamiltonian
    return H

def build_heterogeneous_hamiltonian(k, J, g,ini=0):
    #build hmailtonian acting on k sites starting in site ini
    H = csr_matrix((2**k, 2**k), dtype=complex)

    # Build J * sum sigma^z_i sigma^z_{i+1}
    for i in range(k-1):
        ops = [id2] * k
        ops[i] = sz
        ops[(i + 1)] = sz
        H += J * kron_n(ops)
    

    # Build h * sum sigma^x_i
    for i in range(k):
        ops = [id2] * k
        ops[i] = sx
        H += g[i+ini] * kron_n(ops)

    
    #return the Hamiltonian
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



def non_translation_invariant_sdp(k,L_op,local_obs_array):
    
    #define sdp viariables
    num_rho=L-k+1
    rho_list=[]
    constraints = []

    for i in range(num_rho):
        rho_list.append(cp.Variable((2**k, 2**k), hermitian=True))
        constraints += [rho_list[i] >> 0]
        constraints += [cp.trace(rho_list[i]) == 1]
        
    #impose consistency between overlapping rhos
    for i in range(num_rho - 1):
        constraints += [cp.partial_trace(rho_list[i], [2]*k, 0) == cp.partial_trace(rho_list[i+1], [2]*k, k-1)]
    
    # basis of operators in k-2 internal qubits
    if(k>2):
        X_basis = pauli_matrices.copy()
        
        for _ in range(1, k-2):
            new_basis = []
            for X in X_basis:
                for p in pauli_matrices:
                    new_basis.append(kron(X, p, format='csr'))
            X_basis = new_basis
            
        X_basis_ext=X_basis.copy()
        
        new_basis = []
        for X in X_basis_ext:
            for p in pauli_matrices:
                new_basis.append(kron(X, p, format='csr'))
                
        X_basis_ext = new_basis
        X_basis_ini = X_basis_ext.copy()
        X_basis_fin = X_basis_ext.copy()
        
        for i in range(len(X_basis)):
            X_basis[i] = kron_n([id2,X_basis[i],id2])
        for i in range(len(X_basis_ext)):
            X_basis_ini[i] = kron_n([X_basis_ini[i],id2])
            X_basis_fin[i] = kron_n([id2,X_basis_fin[i]])
            X_basis_ext[i] = kron_n([id2,X_basis_ext[i],id2])
            
    elif(k==2):
        X_basis=[]
        X_basis.append(kron_n([id2,id2]))
        
        new_basis = []
        for p in pauli_matrices:
            new_basis.append(kron(p, format='csr'))
                
        X_basis_ext = new_basis
        X_basis_ini = X_basis_ext.copy()
        X_basis_fin = X_basis_ext.copy()
    
        for i in range(len(X_basis_ext)):
            X_basis_ini[i] = kron_n([X_basis_ini[i],id2])
            X_basis_fin[i] = kron_n([id2,X_basis_fin[i]])
            X_basis_ext[i] = kron_n([id2,X_basis_ext[i],id2])
    
    #build lindbladian constraints
    i=0
    L_op_k = op_support(L_op, i, k, L)
    
    H=build_heterogeneous_hamiltonian(k, J, g_array,i)
    for X in X_basis_ini:
        constraints += [cp.trace(adj_lindblad(X, H, L_op_k,k)@rho_list[i]) == 0]
    
    
    for i in range(1,num_rho-1):
        H=build_heterogeneous_hamiltonian(k, J, g_array,i)
        L_op_k = op_support(L_op, i, k, L)
        for X in X_basis:
            constraints += [cp.trace(adj_lindblad(X, H, L_op_k,k)@rho_list[i]) == 0]
        
    i=num_rho-1
    H=build_heterogeneous_hamiltonian(k, J, g_array,i)
    L_op_k = op_support(L_op, i, k, L)
    for X in X_basis_fin:
        constraints += [cp.trace(adj_lindblad(X, H, L_op_k,k)@rho_list[i]) == 0]
    
    for i in range(num_rho-1):
        
        h_0=build_heterogeneous_hamiltonian(2, J, g_array,i)
        h_dif=build_heterogeneous_hamiltonian(1, J, g_array,i+1)
        h_dif=kron(id2,h_dif)
        h_0=h_0-h_dif
        for i1 in range(k-2):
            h_0=kron(h_0,id2)
        h_1=build_heterogeneous_hamiltonian(k, J, g_array,i+1)
        
        L_op_0= op_support(L_op, i, 1, L)
        for i1 in range(len(L_op_0)):
            L_op_0[i1]=kron(L_op_0[i1],id2)
        L_op_0_more_sites = [term for term in L_op if len(term["sites"]) >= 2]
        L_op_0= L_op_0 + op_support(L_op_0_more_sites, i, 2, L)
        for i1 in range(len(L_op_0)):
            for _ in range(k-2):
                L_op_0[i1]=kron(L_op_0[i1],id2)
                
        L_op_1=op_support(L_op, i+1, k, L)
        
        
        for j in range(len(X_basis_ext)):
            constraints+=[cp.trace(adj_lindblad(X_basis_ini[j], h_1, L_op_1,k)@rho_list[i+1]) + cp.trace(adj_lindblad(X_basis_fin[j], h_0, L_op_0,k)@rho_list[i]) == 0]
    
    #build objective observable
    obs_weight_array=[]
    full_target=0.0
    
    for local_obs in local_obs_array:

        target_q=np.min(local_obs['sites'])
        if target_q > num_rho-1:
            target_q=num_rho-1
        obs=op_support([local_obs], target_q, k, L)[0]
        
        obs_weight_array.append(cp.Parameter(nonneg=True))
        full_target+=obs_weight_array[-1]*obs@rho_list[target_q]

    objective_min=cp.Minimize(cp.real(cp.trace(full_target)))
    prob_min = cp.Problem(objective_min, constraints)
    
    objective_max=cp.Maximize(cp.real(cp.trace(full_target)))
    prob_max = cp.Problem(objective_max, constraints)
    
    
    #solve SDP
    for i in range(len(local_obs_array)):
        for j in range(len(local_obs_array)):
            obs_weight_array[j].value=0.0
        obs_weight_array[i].value=1.0
    
        prob_min.solve(solver=cp.MOSEK, verbose=False,canon_backend=cp.SCIPY_CANON_BACKEND)#set verbose to true to see details on the calculation

        min_obs=prob_min.value
        print('k= ',k)
        print("problem status = ",prob_min.status)
        print("Min of obs: ", min_obs,"\n")
        print("Time:", time.time()-t0)
        
        prob_max.solve(solver=cp.MOSEK, verbose=False,canon_backend=cp.SCIPY_CANON_BACKEND)#set verbose to true to see details on the calculation

        max_obs=prob_max.value
        print('k= ',k)
        print("problem status = ",prob_max.status)
        print("Max of obs: ", max_obs,"\n")
        print("Time:", time.time()-t0)
        
        pos = local_obs_array[i]["sites"][0]

        row = pd.DataFrame([{
            "k": k,
            "min": min_obs,
            "max": max_obs,
            "obs": obs_label,
            "pos": pos
        }])
    
        row.to_csv(filename, mode='a', header=False, index=False)

    
    return 


    
    

'''
Example of a system
'''

#parameters of the system
L=10
k=4
J=0.5
g=0.5
gamma=1
rng = np.random.default_rng(seed=42)
g_array = g + rng.uniform(-0.1, 0.1, size=L)
print("g_array: ", g_array)

#lindbladian operators
L_op=[]
for i in range(L):
    op = np.sqrt(gamma) * 0.5 * (sx - 1j * sy)
    L_current= {
        "ops": [op],
        "sites": [i]
    }
    L_op.append(L_current)


# Directory for results
results_dir = "results_non_ti_random_field"
os.makedirs(results_dir, exist_ok=True)


observables = {
        "sx": sx,
        "sy": sy,
        "sz": sz
    }

for obs_label,obs in observables.items():
    filename = os.path.join(results_dir,f"results_{L}_{obs_label}.csv")

    if not os.path.exists(filename):
        pd.DataFrame(columns=["k", "min", "max", "obs", "pos"]).to_csv(filename, index=False)    

    
    local_obs_array=[]
    for i in range(L):
        local_obs= {
                "ops": [obs],
                "sites": [i]
            }
        local_obs_array.append(local_obs)
        
    
    non_translation_invariant_sdp(k,L_op,local_obs_array)
    t1 = time.time()
    print(f"Tiempo total: {t1 - t0:.2f} segundos")

