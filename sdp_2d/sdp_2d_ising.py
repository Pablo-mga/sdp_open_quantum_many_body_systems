import cvxpy as cp
import numpy as np
from scipy.sparse import identity, csr_matrix
import pandas as pd
import os
import time
from class_operator import Operator,op_support,sx,sy,sz,id2
import pickle
import scipy as sp

t0=time.time()

pauli_matrices=[id2,sx,sy,sz]

def initialize_problem(k):
    #define sdp viariables
    folder_name = f"matrix_constraints_{system}"
    
    c_array = cp.Variable((len(sym_classes), 1))
        

    A = sp.sparse.load_npz(os.path.join(folder_name, 'sym_flattened.npz'))
    
    rho_vec = A @ c_array
    I_flat = identity(2**k, format='csr', dtype=complex).reshape((2**k*2**k, 1), order='C')
    rho_final_vec = (I_flat + rho_vec) / 2**k
    
    rho = cp.reshape(rho_final_vec, (2**k, 2**k), order='C')
    
    constraints_s = [
        rho >> 0,
        cp.trace(rho) == 1
    ]
    
    print("sym constraints imposed: ", time.time()-t0)
        
    g_param=cp.Parameter(nonneg=True)
    sign_param=cp.Parameter()
    
    mat_x = sp.sparse.load_npz(os.path.join(folder_name, 'comm_x_flattened.npz'))
    mat_z = sp.sparse.load_npz(os.path.join(folder_name, 'comm_z_flattened.npz'))
    mat_d = sp.sparse.load_npz(os.path.join(folder_name, 'diss_flattened_ising.npz'))
    
    lindblad_vector = (g_param/2 * mat_x + V/4 * mat_z + gamma * mat_d)
    
    constraints_l = [ lindblad_vector @ cp.reshape(rho, shape = (-1), order = 'C') == 0]
    
    constraints=constraints_s + constraints_l
    
    print("lind constraints imposed: ",time.time()-t0)
    
    objective_min=cp.Minimize(sign_param*cp.real(cp.trace(obs@rho)))
    
    print(time.time()-t0)
    
    prob_min = cp.Problem(objective_min, constraints)
    
    
    return g_param,sign_param,prob_min,rho

'''
Choose system size
'''

# system="2x2"
# region = [[0,0],[0,1],[1,0],[1,1],[2,0]]
# internal_region = [[0,0],[0,1],[1,0],[1,1]]

# system="2x3"
# region = [[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[1,3],[2,0],[2,1]]
# internal_region = [[0,0],[0,1],[0,2],[1,0],[1,1],[1,2]]

system="1x2"
region = [[0,0],[0,1],[0,2],[1,0]]
internal_region = [[0,0],[0,1]]

# system="1x1"
# region = [[0,0],[0,1]]
# internal_region = [[0,0]]


'''
Choose system params
'''
          
gamma=1
V=5.0*gamma




g_array=[]
for i in range(10):
    g_array.append((0.0+0.1*(i))*gamma)
g_last=g_array[-1]

for i in range(30):    
    g_array.append((g_last+0.02*(i+1))*gamma)
g_last=g_array[-1]    
    
for i in range(20):
    g_array.append((g_last+0.2*(i+1))*gamma)    
g_last=g_array[-1]

for i in range(10):
    g_array.append((g_last+0.5*(i+1))*gamma)
g_last=g_array[-1]

for i in range(10):
    g_array.append((g_last+1.0*(i+1))*gamma)

   

'''
sdp
'''

k=len(region)

observables = {
    "sx": Operator(region,[1,]+[0,]*(k-1)),
    "sy": Operator(region,[2,]+[0,]*(k-1)),
    "sz": Operator(region,[3,]+[0,]*(k-1)),
}

results=[]

with open(f"sym_classes{system}.pkl", "rb") as f:
    sym_classes = pickle.load(f)

for obs_label,obs_op in observables.items():
    
    obs=obs_op.to_matrix()

    g_param,sign_param,prob_min,rho=initialize_problem(k)
    
    min_lims=[]
    max_lims=[]
    
    for i,g in enumerate(g_array):
        g_param.value=g
        print("g=",g)
        sign_param.value=1
        
        prob_min.solve(solver=cp.MOSEK, verbose=False,canon_backend=cp.SCIPY_CANON_BACKEND)
        min_obs=prob_min.value
        min_lims.append(min_obs)
        print("problem status = ",prob_min.status)
        print("Min of obs: ", min_obs)
        print("time: ", time.time()-t0,"\n")
        


        sign_param.value=-1
        prob_min.solve(solver=cp.MOSEK, verbose=False,canon_backend=cp.SCIPY_CANON_BACKEND)#set verbose to true to see details on the calculation
        max_obs=-prob_min.value
        max_lims.append(max_obs)
        print("problem status = ",prob_min.status)
        print("Max of obs:", max_obs)
        print("time: ", time.time()-t0,"\n")
        
    results.append(min_lims)
    results.append(max_lims)


minx, maxx, miny, maxy, minz, maxz = results

df = pd.DataFrame({
    "g": g_array,
    "minx": minx,
    "maxx": maxx,
    "miny": miny,
    "maxy": maxy,
    "minz": minz,
    "maxz": maxz
})

folder = "results_ising"
os.makedirs(folder, exist_ok=True)
df.to_csv(f"{folder}/observable_bounds{system}_V_{V:.2f}".replace('.', '_') + ".csv",index=False)


