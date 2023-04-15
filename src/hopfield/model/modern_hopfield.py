import numpy as np

class ModernHopfieldNet:
    def __init__(self, n_units, f = lambda x: x**2):
        self.n_units = n_units
        self.memories = np.empty((0, n_units))
        self.f = f

    def recall(self, s, n_itr = 10):
        for i in range(n_itr):
            self.update(s)
        for i in range(n_itr):
            s = update_method(s)
            if np.array_equal(s, prev_s):
                break
            prev_s = s
        return s
            
    def update(self, s):
        old_s = np.copy(s)
        res = np.matmul(self.memories, old_s.T)
        pos = self.f(res + old_s)
        neg = self.f(res - old_s)
        np.sum(pos - neg, axis = 0)
        for i in range(self.n_units):
            s[:, i] = self.sign()

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
        np.memories = 
    
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