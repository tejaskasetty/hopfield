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
        neuron_idx = np.random.randint(self.n_units)
        s_t[neuron_idx, :] = np.sign(np.matmul(self.W[neuron_idx], s_t))
        return s_t.T

    def store(self, patterns):
        # patterns: n_samples x n_units
        # Compute weight matrix using Hebbian learning rule
        self.W += np.matmul(patterns.T, patterns) / self.n_units
        np.fill_diagonal(self.W, 0)
    
    def delete(self, patterns):
        # patterns: n_samples x n_units
        # Compute weight matrix using Hebbian learning rule
        self.W -= self.matmul(patterns.T, patterns)/ self.n_units
        np.fill_diagonal(self.W, 0)
        
    def energy(self, y):
        # y: batch_size x n_units
        # Compute energy of states
        energy = -0.5 * np.sum(np.matmul(y, self.W) * y, axis=1)

        return energy
    
    def simulate(self, init, n_steps, update ='sync', n_updates = 1):
        traj = [init]
        for i in range(n_steps):
            if update == 'sync':
                new_state = self.sync_update([traj[i]])
            elif update == 'async':
                new_state = self.async_rupdate([traj[i]], n_updates)
            else:
                raise ValueError('Invalid update method. Must be "sync" or "async".')
            traj.append(new_state)
        
        return np.array(traj)