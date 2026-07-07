import numpy as np
from scipy.sparse import kron, csr_matrix, vstack,hstack
import os
import time
from class_operator import Operator,op_support,sx,sy,sz,id2
import pickle
from tqdm import tqdm
import scipy as sp

t0=time.time()

pauli_matrices=[id2,sx,sy,sz]


def build_hamiltonian_x(k):
    ham_list=[]
    for i in range(len(internal_region)):
        op_index=[0]*k
        op_index[i]=1
        ham_list.append(Operator(region,op_index))
        
    #return the list of Hamiltonian terms with sx
    return ham_list

def build_hamiltonian_z(k):
    ham_list = []
    directions = [(1,0), (-1,0), (0,1), (0,-1)]

    for x, y in internal_region:
        for dx, dy in directions:
            sites = [[x, y], [x + dx, y + dy]]
            op = Operator(sites, [3,3])
            if not any(op.is_equal(existing_op) for existing_op in ham_list):
                ham_list.append(op)

    #return the list of Hamiltonian terms with sz sz
    return ham_list

def adj_lindblad_comm_x(X,k):
    comm=csr_matrix((2**k, 2**k), dtype=complex)
    X_matrix=X.to_matrix()
    ham_list=build_hamiltonian_x(k)
    for op in ham_list:
        op_mat=op.to_matrix()
        comm += 1j * (op_mat @ X_matrix - X_matrix @ op_mat)
    return comm

def adj_lindblad_comm_z(X,k):
    comm=csr_matrix((2**k, 2**k), dtype=complex)
    ham_list=build_hamiltonian_z(k)
    for op in ham_list:
    
        extended_region = internal_region + [s for s in op.region if s not in internal_region]
        op_idx=[]
        for i in range(len(extended_region)):
            op_idx.append(i+1)
            
        mock_op = Operator(extended_region, op_idx)
        
        fitted_op,internal_change=mock_op.find_op_inside_region(region)
        X_aux=Operator(X.region.copy(),X.op_idx.copy())
        
        X_aux = X_aux.transform_operator(internal_change, region)
        op = op.transform_operator(internal_change, region)

        X_matrix=X_aux.to_matrix()
        op_mat=op.to_matrix()
        
        comm += 1j * (op_mat @ X_matrix - X_matrix @ op_mat)
    return comm

def adj_lindblad_diss(X_op,L_op_dict_array,k):
    dissipator = csr_matrix((2**k, 2**k), dtype=complex)
    
    for L_op_dict in L_op_dict_array:
        
        X_aux=Operator(X_op.region.copy(),X_op.op_idx.copy())
        internal_change=L_op_dict["internal_change"]
        X_aux = X_aux.transform_operator(internal_change,region)
        X_matrix=X_aux.to_matrix()
        
        L_op=op_support(L_op_dict, k)
        L_op_d=L_op.getH()
        
        dissipator+= L_op_d @ X_matrix @ L_op -0.5*(L_op_d @ L_op @ X_matrix + X_matrix @ L_op_d @ L_op)
    
    return dissipator



def initialize_lindblad_operators_dicke(gamma):

    local_lindblad =  0.5*(sx - 1j * sy)
    local_lindblad = np.sqrt(gamma)*(kron(local_lindblad,id2)+kron(id2,local_lindblad))
    
    L_op_dict_array=[]
    
    for x,y in internal_region:
        for i in [-1,1]:
            sites=sorted([[x,y],[x + i,y]])
            lindblad_dict={
                "ops": local_lindblad,
                "sites": sites
                }
            if lindblad_dict not in L_op_dict_array:
                L_op_dict_array.append(lindblad_dict)
            
            sites=sorted([[x,y],[x ,y+i]])
            lindblad_dict={
                "ops": local_lindblad,
                "sites": sites 
                }
            if lindblad_dict not in L_op_dict_array:
                L_op_dict_array.append(lindblad_dict)
                
                
    
    for L_op_dict in L_op_dict_array:
        
        extended_region = internal_region + [s for s in L_op_dict["sites"] if s not in internal_region]
        op_idx=[]
        for i in range(len(extended_region)):
            op_idx.append(i+1)
            
        mock_op = Operator(extended_region, op_idx)
        
        fitted_op,internal_change=mock_op.find_op_inside_region(region)
        
        
        old_pos = {label: mock_op.region[i] for i, label in enumerate(mock_op.op_idx)}
        new_pos = {label: fitted_op.region[i] for i, label in enumerate(fitted_op.op_idx)}
        
        L_op_dict["sites"] = [new_pos[label] for label,site in old_pos.items() if site in L_op_dict["sites"]]
        L_op_dict["internal_change"] = internal_change
        L_op_dict["sites_idx"] = [region.index(site) for site in L_op_dict["sites"]]
        
        
    return L_op_dict_array

def initialize_lindblad_operators_ising(gamma):

    local_lindblad =  0.5*(sx - 1j * sy)
    local_lindblad = np.sqrt(gamma)*local_lindblad
    
    L_op_dict_array=[]
    
    for x,y in internal_region:
        sites=[[x,y]]
        lindblad_dict={
            "ops": local_lindblad,
            "sites": sites
            }
        L_op_dict_array.append(lindblad_dict)
                
    
    for L_op_dict in L_op_dict_array:
        
        extended_region = internal_region + [s for s in L_op_dict["sites"] if s not in internal_region]
        op_idx=[]
        for i in range(len(extended_region)):
            op_idx.append(i+1)
            
        mock_op = Operator(extended_region, op_idx)
        
        fitted_op,internal_change=mock_op.find_op_inside_region(region)
        
        
        old_pos = {label: mock_op.region[i] for i, label in enumerate(mock_op.op_idx)}
        new_pos = {label: fitted_op.region[i] for i, label in enumerate(fitted_op.op_idx)}
        
        L_op_dict["sites"] = [new_pos[label] for label,site in old_pos.items() if site in L_op_dict["sites"]]
        L_op_dict["internal_change"] = internal_change
        L_op_dict["sites_idx"] = [region.index(site) for site in L_op_dict["sites"]]
        
        
    return L_op_dict_array



'''
Choose system size
'''

# system="2x2"
# region = [[0,0],[0,1],[1,0],[1,1],[2,0]]
# internal_region = [[0,0],[0,1],[1,0],[1,1]]

# system="2x3"
# region = [[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[1,3],[2,0],[2,1]]
# internal_region = [[0,0],[0,1],[0,2],[1,0],[1,1],[1,2]]

# system="1x2"
# region = [[0,0],[0,1],[0,2],[1,0]]
# internal_region = [[0,0],[0,1]]

system="1x1"
region = [[0,0],[0,1]]
internal_region = [[0,0]]

# system="1x4"
# region = [[0,0],[0,1],[0,2],[0,3],[0,4],[1,0],[1,1]]
# internal_region= [[0,0],[0,1],[0,2],[0,3]]

# system="3x3"
# region = [[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2],[3,1],[3,2]]
# internal_region = [[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2]]




   

'''
sdp
'''

gamma=1

k=len(region)
L_op_dict_array_dicke=initialize_lindblad_operators_dicke(gamma)
L_op_dict_array_ising=initialize_lindblad_operators_ising(gamma)


with open(f"sym_classes{system}.pkl", "rb") as f:
    sym_classes = pickle.load(f)


class_rep_op_array=[]

print('Reading symmetry classes')
for sym_class in tqdm(sym_classes):
    

    for op in sym_class:

        if op.check_operator_inside(internal_region):

            op_copy = Operator(op.region.copy(), op.op_idx.copy())
            op_copy.change_region(region)

            class_rep_op_array.append(op_copy)
            break

#lindblad constraints
comm_x_rows = []
comm_z_rows = []
diss_rows_dicke = []
diss_rows_ising = []

print('Generating lindblad constraints')
for X in tqdm(class_rep_op_array):
    comm_x_rows.append(adj_lindblad_comm_x(X, k).reshape((1, 4**k), order='F'))
    comm_z_rows.append(adj_lindblad_comm_z(X, k).reshape((1, 4**k), order='F'))
    diss_rows_dicke.append(adj_lindblad_diss(X, L_op_dict_array_dicke, k).reshape((1, 4**k), order='F'))
    diss_rows_ising.append(adj_lindblad_diss(X, L_op_dict_array_ising, k).reshape((1, 4**k), order='F'))

mat_x = vstack(comm_x_rows)
mat_z = vstack(comm_z_rows)
mat_d_dicke = vstack(diss_rows_dicke)
mat_d_ising = vstack(diss_rows_ising)

folder_name = f"matrix_constraints_{system}"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

sp.sparse.save_npz(os.path.join(folder_name, 'comm_x_flattened.npz'), mat_x)
sp.sparse.save_npz(os.path.join(folder_name, 'comm_z_flattened.npz'), mat_z)
sp.sparse.save_npz(os.path.join(folder_name, 'diss_flattened_dicke.npz'), mat_d_dicke)
sp.sparse.save_npz(os.path.join(folder_name, 'diss_flattened_ising.npz'), mat_d_ising)


#symmetry constraints
print('Generating symmetry constraints')
flattened_matrices = []
for i, sym_class in enumerate(tqdm(sym_classes)):
    class_op_sum = csr_matrix((2**k, 2**k), dtype=complex)        
    
    for op in sym_class:
        class_op_sum += op.to_matrix()
    
    flattened_matrices.append(class_op_sum.reshape((2**k*2**k, 1), order='C'))
mat_sym = hstack(flattened_matrices)
sp.sparse.save_npz(os.path.join(folder_name, 'sym_flattened.npz'), mat_sym)




