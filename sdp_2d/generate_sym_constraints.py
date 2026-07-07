import numpy as np
from scipy.sparse import identity, csr_matrix
import time
import itertools
from class_operator import Operator
import pickle


t0=time.time()

# Define Pauli matrices
sx = csr_matrix(np.array([[0, 1], [1, 0]], dtype=complex))
sy = csr_matrix(np.array([[0, -1j], [1j, 0]], dtype=complex))
sz = csr_matrix(np.array([[1, 0], [0, -1]], dtype=complex))
id2 = identity(2, format='csr', dtype=complex)

pauli_matrices=[id2,sx,sy,sz]

'''
Choose system
'''

# system="3x3"
# region = [[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2],[3,1],[3,2]]

# system="2x3"
# region = [[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[1,3],[2,0],[2,1]]

# system="1x2"
# region = [[0,0],[0,1],[0,2],[1,0]]

# system="2x2"
# region = [[0,0],[0,1],[1,0],[1,1],[2,0]]

system="1x1"
region = [[0,0],[0,1]]


'''
generate constraints
'''

k=len(region)


sym_classes=[]
for num_sites in range(1,k+1):
    
    labels = [1,2,3]
    
    operators = []
    
    for sites in itertools.combinations(range(len(region)), num_sites):      
        for ops in itertools.product(labels, repeat=num_sites):    
            
            op_idx = [0]*len(region)
            for i,site in enumerate(sites):
                op_idx[site] = ops[i]
    
            operators.append(op_idx)
            
    visited = [False] * len(operators)
    op_to_index = {tuple(op): i for i, op in enumerate(operators)}
    
    sym_classes_num_sites=[]
    
    for i, op_index in enumerate(operators):

        if visited[i]:
            continue

        op = Operator(region, op_index)
        current_sym = op.symmetry_class()
        sym_classes_num_sites.append(current_sym)

        for other_op in current_sym:
            idx = op_to_index.get(tuple(other_op.op_idx))
            visited[idx] = True

    print('sites considered: ',num_sites)
    
    sym_classes += sym_classes_num_sites
    
    print(time.time() - t0)
    
print('Number of symmetry classes: ',len(sym_classes))

with open(f"sym_classes{system}.pkl", "wb") as f:
    pickle.dump(sym_classes, f)
    

    
 