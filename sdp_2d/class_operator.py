import numpy as np
from scipy.sparse import kron, identity, csr_matrix
import sparse

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

class Operator:
    def __init__(self, region, op_idx):
        if(len(region)==len(op_idx)):
            self.region = region
            self.op_idx = op_idx
            self.lx = max([x for x,y in region])-min([x for x,y in region]) + 1
            self.ly = max([y for x,y in region])-min([y for x,y in region]) + 1
        else:
            raise ValueError("Operator and region sites do not match.")
            
    
    def __str__(self):
        return f"Region: {self.region}, Operator indices: {self.op_idx}"
    
    def translate(self, dx,dy):
        new_region = [[x+dx,y+dy] for x,y in self.region]
        return Operator(new_region, self.op_idx)
    
    def check_operator_inside(self, region):
        sites= [self.region[i] for i in range(len(self.op_idx)) if self.op_idx[i] != 0]
        for site in sites:
            if site not in region:
                return False
        return True
    
    def change_region(self, new_region):
        if self.check_operator_inside(new_region):
            new_idx = [0,]*len(new_region)
            for i, site in enumerate(new_region):
                if site in self.region:
                    idx = self.region.index(site)
                    new_idx[i] = self.op_idx[idx]
            self.region = new_region
            self.op_idx = new_idx
        else:
            raise ValueError("Operator cannot be changed to the new region as it is outside the region.")

    def is_equal(self, other):
        if other.check_operator_inside(self.region):
            other_copy = Operator(other.region.copy(), other.op_idx.copy())
            other_copy.change_region(self.region)
            return self.op_idx == other_copy.op_idx
        return False
    
    def to_matrix(self):
        ops_list = [pauli_matrices[idx] for idx in self.op_idx]
        return kron_n(ops_list)
    
    def rotate(self,angle):
        if angle == 0:
            return Operator(self.region, self.op_idx)
        elif angle == 1:
            new_region = [[y,-x] for x,y in self.region]
            return Operator(new_region, self.op_idx)
        elif angle == 2:
            new_region = [[-x,-y] for x,y in self.region]
            return Operator(new_region, self.op_idx)
        elif angle == 3:
            new_region = [[-y,x] for x,y in self.region]
            return Operator(new_region, self.op_idx)
    
    def reflect(self,axis=0):
        if axis == 0:
            new_region = [[x,-y] for x,y in self.region]
            return Operator(new_region, self.op_idx)
        elif axis == 1:
            new_region = [[-x,y] for x,y in self.region]
            return Operator(new_region, self.op_idx)
    
    def symmetry_class(self):
        sym_ops = []
        sym_ops.append(self)

        for angle in [0,1,2,3]:
            
            new_op = self.rotate(angle)
            for i in range(-self.lx,self.lx):
                for j in range(-self.ly,self.ly):
                    new_op_translated = new_op.translate(i,j)
                    if new_op_translated.check_operator_inside(self.region):
                        if not any(op.is_equal(new_op_translated) for op in sym_ops):
                            new_op_translated.change_region(self.region)
                            sym_ops.append(new_op_translated)      
            

        ref_op=self.reflect(0)
        for angle in [0,1,2,3]:
            new_op = ref_op.rotate(angle)
            for i in range(-self.lx,self.lx):
                for j in range(-self.ly,self.ly):
                    new_op_translated = new_op.translate(i,j)
                    if new_op_translated.check_operator_inside(self.region):
                        if not any(op.is_equal(new_op_translated) for op in sym_ops):
                            new_op_translated.change_region(self.region)
                            sym_ops.append(new_op_translated)

        return sym_ops
    
    def find_op_inside_region(self,region):
        for angle in [0,1,2,3]:
    
            new_op = self.rotate(angle)
    
            for i in range(0, self.lx):
                for j in range(0, self.ly):
    
                    new_op_translated = new_op.translate(i, j)    
                    if new_op_translated.check_operator_inside(region):
                        new_op_translated.change_region(region)
                        return new_op_translated,[0,angle,i,j]
                    
            for i in range(0, self.lx):
                for j in range(-self.ly, 0):
    
                    new_op_translated = new_op.translate(i, j)    
                    if new_op_translated.check_operator_inside(region):
                        new_op_translated.change_region(region)
                        return new_op_translated,[0,angle,i,j]
            
            for i in range(-self.lx, 0):
                for j in range(-self.ly, self.ly):
    
                    new_op_translated = new_op.translate(i, j)    
                    if new_op_translated.check_operator_inside(region):
                        new_op_translated.change_region(region)
                        return new_op_translated,[0,angle,i,j]
    
        ref_op = self.reflect(0)
    
        for angle in [0,1,2,3]:
    
            new_op = ref_op.rotate(angle)
    
            for i in range(-self.lx, self.lx):
                for j in range(-self.ly, self.ly):
    
                    new_op_translated = new_op.translate(i, j)
    
                    if new_op_translated.check_operator_inside(region):
                        new_op_translated.change_region(region)
                        return new_op_translated,[1,angle,i,j]

        raise ValueError("No transformation places operator inside the region")
        
    def is_symmetry_related(self, other):
        for angle in [0,1,2,3]:
            new_op = self.rotate(angle)
            for i in range(-self.lx, self.lx):
                for j in range(-self.ly, self.ly):
                    new_op_translated = new_op.translate(i, j)
                    if new_op_translated.check_operator_inside(other.region):
                        new_op_translated.change_region(other.region)
                        if new_op_translated.op_idx == other.op_idx:
                            return True

        ref_op = self.reflect(0)
        for angle in [0,1,2,3]:
            new_op = ref_op.rotate(angle)
            for i in range(-self.lx, self.lx):
                for j in range(-self.ly, self.ly):
                    new_op_translated = new_op.translate(i, j)
                    if new_op_translated.check_operator_inside(other.region):
                        new_op_translated.change_region(other.region)
                        if new_op_translated.op_idx == other.op_idx:
                            return True

        return False
    
    def transform_operator(self, transformation, region):
        op = self
        if transformation[0] == 1:
            op = op.reflect()
        op = op.rotate(transformation[1])
        op = op.translate(transformation[2], transformation[3])
        op.change_region(region)
        return op
        
        
def op_support(local_op, k ):
    """
    Build operators acting on qubits
    ini, ..., ini+k-1.

    Parameters
    ----------
    L_op : dict
        [{"ops": csr_matrix (2**m x 2**m), "sites": [i1,i2,...,im]}]
    k : int
        number of qubits to be considered

    Returns
    -------
    csr_matrix that can completely act on the qubits
        operators 2^k x 2^k
    """

    op = local_op["ops"]
    sites = local_op["sites_idx"]
    
    m = len(sites)
    if op.shape != (2**m, 2**m):
        raise ValueError("Operator size incompatible with number of sites")
    

    if all( 0<= ls < k for ls in sites):
        full_op = kron(op, kron_n([id2]*(k-m))) if k > m else op
    else:
        raise ValueError("Sites fuera de rango")
                
    s = sparse.COO.from_scipy_sparse(full_op)
    
    T = s.reshape([2] * (2 * k),order="C")
    
    
    perm = sites + [i for i in range(k) if i not in sites]
    inv_perm = np.argsort(perm)
    axes = list(inv_perm) + [p + k for p in inv_perm]
    
    
    T_perm = T.transpose(axes)
    final_sparse = T_perm.reshape((2**k, 2**k),order="C")
    
    return final_sparse.to_scipy_sparse().tocsr()

 