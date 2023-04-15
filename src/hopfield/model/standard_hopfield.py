import numpy as np

class HopfieldNet:
    def __init__(self, n_units):
        self.n_units = n_units
        self.W = np.zeros((n_units, n_units), dtype = 'float64') #weight matrix

    def recall(self, s, n_itr = 10, update = 'sync'):
        update_method =  self.async_update if update == 'async' else self.sync_update
        prev_s = s
        for i in range(n_itr):
            s = update_method(s)
            if np.array_equal(s, prev_s):
                break
            prev_s = s
        return s
            
    def sync_update(self, s):
        # update all the neurons at the same time
        h = np.matmul(s, self.W)
        s = np.sign(h)
        return s
    
    def async_update(self, s):
        # sequential neuron update
        s_t = np.copy(s.T)
        for i in range(self.n_units):
            s_t[i, :] = np.sign(np.matmul(self.W[i], s_t))
        return s_t.T
    
    def async_rupdate(self, s, n_updates = 1):
        # random neuron update
        s_t = np.copy(s.T)
        idx = np.random.randint(self.n_units)
        s_t[idx] = np.sign(np.matmul(self.W[idx], s_t))
        return s_t.T

    def store(self, patterns):
        # patterns: n_samples x n_units
        # Compute weight matrix using Hebbian learning rule
        hebb_update = np.matmul(patterns.T, patterns)/ self.n_units
        self.W += hebb_update
        np.fill_diagonal(self.W, 0)
    
    def delete(self, patterns):
        # patterns: n_samples x n_units
        # Compute weight matrix using Hebbian learning rule
        hebb_update = np.matmul(patterns.T, patterns)/ self.n_units
        self.W -= hebb_update
        np.fill_diagonal(self.W, 0)
        
    def energy(self, y):
        # y: batch_size x n_units
        # Compute energy of states
        energy = -0.5 * np.sum(np.matmul(y, self.W) * y, axis=1)

        return energy
    
    def simulate(self, init, n_steps, update ='async', n_updates = 1):
        # simulate trajectory of partial memory recall
        traj = [init]
        for i in range(n_steps):
            if update == 'sync':
                new_state = self.sync_update(traj[i])
            elif update == 'async':
                new_state = self.async_rupdate(traj[i], n_updates)
            else:
                raise ValueError('Invalid update method. Must be "sync" or "async".')
            traj.append(new_state)
        
        return traj
    
class SmallWorldHopfieldNet(HopfieldNet):
    def __init__(self, n_units, cluster_coeff, p):
        super().__init__(n_units)
        self.cluster_coeff = cluster_coeff
        self.k = int(cluster_coeff * n_units / 2)
        self.p = p
        self.W =  np.zeros((n_units, n_units))
        self.lattice = self.__gen_small_world_lattice()
    
    def store(self, patterns):
        # patterns: n_samples x n_units
        # call to the standard hopfield store
        super().store(patterns)
        
        # Apply small-world topology to weight matrix
        self.apply_small_world_topology()
    
    def delete(self, patterns):
        # patterns: n_samples x n_units
        # call to the standard hopfield delete
        super().delete(patterns)
        
        # Apply small-world topology to weight matrix
        self.apply_small_world_topology()
    
    
    def apply_small_world_topology(self):
        # Apply connections to weight matrix
        self.W *= self.lattice
    
    def sparsity(self):
        return np.sum(self.lattice)/(self.n_units * self.n_units)
    
    def __gen_small_world_lattice(self):
        # Create regular lattice
        lattice = np.zeros((self.n_units, self.n_units))
        for i in range(self.n_units):
            for j in range(1, self.k+1):
                lattice[i, (i+j) % self.n_units] = 1
                lattice[i, (i-j) % self.n_units] = 1

        # Add and remove random connection 
        for i in range(self.n_units):
            for j in range(i+1, self.n_units):
                if np.random.binomial(1, self.p):
                    dist = min(abs(j-i), self.n_units - abs(j-i))
                    if dist <= self.k and np.random.uniform() < self.p:
                        # Remove close connection
                        lattice[i, j] = 0
                        lattice[j, i] = 0
                    elif dist > self.k and np.random.uniform() < self.p:
                        # Add distant connection
                        lattice[i, j] = 1
                        lattice[j, i] = 1
        return lattice
