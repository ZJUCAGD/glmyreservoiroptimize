import numpy as np
from minBasis import minBasis
from PPH import persHomo
import gudhi as gd
import networkx as nx
from matplotlib import pyplot as plt


def TDE(sequence, d, k=5):
    sequence = np.array(sequence)
    length = len(sequence)

    max_i = length - (d - 1) * k
    if max_i <= 0:
        raise ValueError("Sequence length insufficient for time delay embedding. Please reduce d or k, or increase input sequence length.")

    result = np.zeros((max_i, d))

    for i in range(max_i):
        indices = i + k * np.arange(d)
        result[i, :] = sequence[indices]

    return result

def mean_cosine_columns(A: np.ndarray) -> float:
    """
    Calculate the average cosine of angles between all pairs of column vectors in square matrix A.
    If a column is all zeros, its cosine with any other column is treated as 1.
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Input must be an n×n 2D square matrix")

    n = A.shape[1]
    if n < 2:
        return 1.0

    is_zero_col = ~np.any(A, axis=0)

    norms = np.linalg.norm(A, axis=0)
    norms[norms == 0] = 1
    A_norm = A / norms

    cos_mat = A_norm.T @ A_norm

    np.fill_diagonal(cos_mat, 0)

    if np.any(is_zero_col):
        cos_mat[:, is_zero_col] = 1
        cos_mat[is_zero_col, :] = 1
        np.fill_diagonal(cos_mat, 0)

    triu_sum = np.abs(np.triu(cos_mat, k=1).sum())
    mean_cos = triu_sum / (n * (n - 1) / 2)
    print(mean_cos)
    return mean_cos

def memory_capacity(input_series, output_series, k_max=None):
    """
    Calculate total Memory Capacity (MC)

    Args:
        input_series   (np.ndarray): Input time series (T,)
        output_series  (np.ndarray): Output time series (T,)
        k_max          (int or None): Maximum delay steps, defaults to output_series.shape[0]

    Returns:
        float: Total memory capacity MC
        list:  MC_k value for each delay k
    """
    T = len(input_series)
    
    if k_max is None:
        k_max = T - 1
    
    if k_max <= 0 or k_max >= T:
        raise ValueError("k_max must be a positive integer and less than input length")

    mc_values = []
    
    for k in range(1, k_max + 1):
        u = input_series[:-k]
        y = output_series[k:]

        covariance = np.cov(u, y)[0, 1]
        var_u = np.var(u)
        var_y = np.var(y)

        if var_u == 0 or var_y == 0:
            mc_k = 0.0
        else:
            mc_k = (covariance ** 2) / (var_u * var_y)

        mc_values.append(mc_k)

    total_mc = sum(mc_values)
    return total_mc, mc_values

def build_small_world_W(n: int,
                        k: int = 10,
                        beta: float = 0.3,
                        seed: int | None = None) -> np.ndarray:
    """
    Generate a n×n small-world network weight matrix (Watts-Strogatz)
    """
    assert k % 2 == 0 and k < n  #`k` must be an even number and less than `n`
    rng = np.random.RandomState(seed)

    # Unidirectional watts–strogatz graph
    G = nx.watts_strogatz_graph(n, k, beta, seed=seed)
    G = nx.DiGraph((u, v) for u, v in G.edges())
    W = nx.to_numpy_array(G, dtype=float)
    
    # Ensure no bidirectional edges; use a few self-loops to ensure stability
    for i in range(n):
        for j in range(n):
            if W[i,j]!=0:
                W[i,j]=1
    for i in range(n):
        W[i,i]=1 if np.random.RandomState(i).rand() < 0.1 else 0
        for j in range(i):
            W[i,j]=0

    return W

def build_scale_free_W(n: int,
                       m: int = 8,
                       seed: int | None = None) -> np.ndarray:
    """
    Generate a n×n scale-free network weight matrix (Barabási–Albert)
    """
    rng = np.random.RandomState(seed)
    G_undirected = nx.barabasi_albert_graph(n, m, seed=seed)
    G = nx.DiGraph((u, v) if rng.rand() <0.1 else (v, u) for u, v in G_undirected.edges())
    W = nx.to_numpy_array(G, dtype=float)
    
    # Ensure no bidirectional edges; use a few self-loops to ensure stability
    for i in range(n):
        for j in range(n):
            if W[i,j]!=0:
                W[i,j]=1
    for i in range(n):
        W[i,i]=1 if np.random.RandomState(i).rand() < 0.1 else 0
        for j in range(i):
            W[i,j]=0
    
    return W

def Initial_random_W(W: np.ndarray,
                   seed: int | None = None) -> np.ndarray:
    """
    Generate a n×n random network weight matrix
    """
    rng = np.random.RandomState(seed)
    n=W.shape[0]
    
    # Ensure no bidirectional edges; use a few self-loops to ensure stability
    for i in range(n):
        for j in range(n):
            if W[i,j]!=0:
                W[i,j]=1
    for i in range(n):
        W[i,i]=1 if np.random.RandomState(i).rand() < 0.1 else 0
        for j in range(i):
            W[i,j]=0
    
    return W

def normalize_to_range(arr):
    # Step 1: Find min and max of the array
    arr_min = arr.min()
    arr_max = arr.max()

    # Step 2: Normalize to [0, 1]
    arr_normalized = (arr - arr_min) / (arr_max - arr_min)
    
    # Step 3: Scale to [-0.5, 0.5]
    arr_scaled = arr_normalized - 0.5
    
    return arr_scaled

def convert_directed_to_undirected(W):
    n = len(W)
    adj_list = [[] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if W[i][j] != 0 or W[j][i] != 0:
                if j not in adj_list[i]:
                    adj_list[i].append(j)
                if i not in adj_list[j]:
                    adj_list[j].append(i)

    return adj_list

def find_shortest_cycles(adj_list):
    n = len(adj_list)
    min_cycle_length = float('inf')
    minimal_cycles = set()

    def dfs(start, current, visited, path):
        nonlocal min_cycle_length
        visited[current] = True
        path.append(current)

        for neighbor in adj_list[current]:
            if not visited[neighbor]:
                dfs(start, neighbor, visited, path)
            elif neighbor == start and len(path) > 2:
                cycle = path[:]
                min_index = cycle.index(min(cycle))
                normalized_cycle = tuple(cycle[min_index:] + cycle[:min_index])
                
                if len(normalized_cycle) < min_cycle_length:
                    min_cycle_length = len(normalized_cycle)
                    minimal_cycles.clear()
                    minimal_cycles.add(normalized_cycle)
                elif len(normalized_cycle) == min_cycle_length:
                    minimal_cycles.add(normalized_cycle)

        path.pop()
        visited[current] = False

    for start_node in range(n):
        visited = [False] * n
        dfs(start_node, start_node, visited, [])

    return [list(cycle) for cycle in minimal_cycles]

def not_visit_num(loop,visit):
    num=0
    visited=[]
    for p in loop:
        if (p[0],p[1]) in visit:
            num+=1
            visited.append((p[0],p[1]))
        if (p[1],p[0]) in visit:
            num+=1
            visited.append((p[0],p[1]))
    return num,visited

def create_cycle(edges, start=None):
    edgesc = set(edges)
    if start is None:
        start = edgesc.pop()
    cycle = [start[0], start[1]]

    while edgesc:
        for edge in edgesc:
            if edge[0] == cycle[-1]:
                cycle.append(edge[1])
                edgesc.remove(edge)
                break
            elif edge[1] == cycle[-1]:
                cycle.append(edge[0])
                edgesc.remove(edge)
                break

    return cycle[:-1]

def find_directed_cycle(nodes, directed_edges):
    from collections import defaultdict, deque
    
    n = len(nodes)
    node_index = {node: i for i, node in enumerate(nodes)}
    
    out_degree = defaultdict(int)
    in_degree = defaultdict(int)
    
    # Step 1: Initialize directed edges
    directed_set = set()
    for u, v in directed_edges:
        directed_set.add((u, v))
        out_degree[u] += 1
        in_degree[v] += 1
    
    # Step 2: Check initial feasibility
    for node in nodes:
        if out_degree[node] > 1 or in_degree[node] > 1:
            return None
    
    # Step 3: Attempt to orient remaining edges
    cycle = []
    visited = set()
    
    def dfs(node):
        if node in visited:
            return
        visited.add(node)
        cycle.append(node)
        
        for neighbor in [nodes[(node_index[node] + 1) % n]]:
            if (node, neighbor) not in directed_set and (neighbor, node) not in directed_set:
                directed_set.add((node, neighbor))
                out_degree[node] += 1
                in_degree[neighbor] += 1
            if out_degree[node] == 1 and in_degree[neighbor] == 1:
                dfs(neighbor)
    
    start_node = nodes[0]
    dfs(start_node)
    
    # Step 4: Verify if all nodes form a single cycle
    if len(cycle) == n and all(out_degree[node] == 1 and in_degree[node] == 1 for node in nodes):
        #print(cycle)
        return cycle
    else:
        return None


def generate_squares(length,width,allb=False):
    mat=np.zeros((length*width,length*width))
    for xi in range(length):
        for yi in range(width):
            i=width*xi+yi
            
            if xi+1<length:
                xj=xi+1
                yj=yi
                j=width*xj+yj
                if allb:
                    mat[i,j]=1
                else:
                    mat[i,j]=np.random.choice([-1, 1])
            if yi+1<width:
                yj=yi+1
                xj=xi
                j=width*xj+yj
                if allb:
                    mat[i,j]=1
                else:
                    mat[i,j]=np.random.choice([-1, 1])
    return mat

def compute_PPH(mat):
    G=nx.DiGraph()
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i,j]!=0:
                G.add_edge(i,j, weight=mat[i,j])

    g=persHomo(G)
    g.perHom(10)
    g_pair = np.array(g.pair)
    #print(g_pair)
    minB=minBasis(g)
    minB.ComputeAnnotation()
    minB.ComputeMinimalBasis()

    #gd.plot_persistence_diagram(g_pair)
    #plt.gcf().set_size_inches(8, 6) 
    #plt.show()
    return g_pair,minB.MinHomoBasis

def modify_ESNW(mat):
    """
    Modify directed graph adjacency matrix to increase the number of directed cycles
    
    Returns:
        Modified matrix mat
    """
    cycle_modifications = 0
    
    g_pair, basis = compute_PPH(mat)
    modified_edges = set()
    modified_cycles = []
    
    for loop in basis:
        visit_num, visited = not_visit_num(loop, modified_edges)
        
        if len(loop) == 3:
            for p in loop:
                modified_edges.add((p[0], p[1]))
            continue
        
        if len(loop) > 3 and visit_num == 0:
            cycle = create_cycle(loop)
            
            can_modify = True
            edges_to_modify = []
            
            for u, v in zip(cycle, cycle[1:] + cycle[:1]):
                if (u, v) in modified_edges or (v, u) in modified_edges:
                    can_modify = False
                    break
                edges_to_modify.append((u, v))
            
            if can_modify:
                for u, v in edges_to_modify:
                    if mat[u, v] == 0 and mat[v, u] != 0:
                        mat[u, v] = mat[v, u]
                        mat[v, u] = 0
                    elif mat[u, v] != 0 and mat[v, u] == 0:
                        pass
                    elif mat[u, v] != 0 and mat[v, u] != 0:
                        pass
                    
                    modified_edges.add((u, v))
                
                modified_cycles.append(cycle)
                cycle_modifications += 1
    
    print(f"Add {cycle_modifications} cycles!")
    return mat,cycle_modifications

def delete_nodes(W, visit):
    W = np.array(W)
    num_nodes = W.shape[0]
    
    nodes_to_delete = set()
    for node in range(num_nodes):
        outgoing_edges = [(node, neighbor) for neighbor in range(num_nodes) if W[node][neighbor] != 0]
        incoming_edges = [(neighbor, node) for neighbor in range(num_nodes) if W[neighbor][node] != 0]
        
        if all(edge in visit for edge in outgoing_edges + incoming_edges):
            nodes_to_delete.add(node)
    
    print("Nodes to delete:", nodes_to_delete)
    for node in nodes_to_delete:
        outgoing_edges = [(node, neighbor) for neighbor in range(num_nodes) if W[node][neighbor] != 0]
        incoming_edges = [(neighbor, node) for neighbor in range(num_nodes) if W[neighbor][node] != 0]
        print(f"Node {node} outgoing edges: {outgoing_edges}")
        print(f"Node {node} incoming edges: {incoming_edges}")
    
    for node in nodes_to_delete:
        W[node, :] = 0
        W[:, node] = 0
    
    return W

def correct_dimensions(s, targetlength):
    """checks the dimensionality of some numeric argument s, broadcasts it
       to the specified length if possible.

    Args:
        s: None, scalar or 1D array
        targetlength: expected length of s

    Returns:
        None if s is None, else numpy vector of length targetlength
    """
    if s is not None:
        s = np.array(s)
        if s.ndim == 0:
            s = np.array([s] * targetlength)
        elif s.ndim == 1:
            if not len(s) == targetlength:
                raise ValueError("arg must have length " + str(targetlength))
        else:
            raise ValueError("Invalid argument")
    return s


def identity(x):
    return x


'''
Copyright (c) 2015 Clemens Korndörfer

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

class ESN():
    def __init__(self, n_inputs, n_outputs, n_reservoir=200,
                 spectral_radius=0.95, sparsity=0, noise=0.001, input_shift=None,
                 input_scaling=None, teacher_forcing=True, feedback_scaling=None,
                 teacher_scaling=None, teacher_shift=None,
                 out_activation=identity, inverse_out_activation=identity,
                 random_state=None, silent=True):
        """
        Args:
            n_inputs: nr of input dimensions
            n_outputs: nr of output dimensions
            n_reservoir: nr of reservoir neurons
            spectral_radius: spectral radius of the recurrent weight matrix
            sparsity: proportion of recurrent weights set to zero
            noise: noise added to each neuron (regularization)
            input_shift: scalar or vector of length n_inputs to add to each
                        input dimension before feeding it to the network.
            input_scaling: scalar or vector of length n_inputs to multiply
                        with each input dimension before feeding it to the netw.
            teacher_forcing: if True, feed the target back into output units
            teacher_scaling: factor applied to the target signal
            teacher_shift: additive term applied to the target signal
            out_activation: output activation function (applied to the readout)
            inverse_out_activation: inverse of the output activation function
            random_state: positive integer seed, np.rand.RandomState object,
                          or None to use numpy's builting RandomState.
            silent: supress messages
        """
        # check for proper dimensionality of all arguments and write them down.
        self.data_input = None
        self.n_inputs = n_inputs
        self.n_reservoir = n_reservoir
        self.n_outputs = n_outputs
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.noise = noise
        self.input_shift = correct_dimensions(input_shift, n_inputs)
        self.input_scaling = correct_dimensions(input_scaling, n_inputs)

        self.teacher_scaling = teacher_scaling
        self.teacher_shift = teacher_shift

        self.out_activation = out_activation
        self.inverse_out_activation = inverse_out_activation
        self.random_state = random_state

        # the given random_state might be either an actual RandomState object,
        # a seed or None (in which case we use numpy's builtin RandomState)
        if isinstance(random_state, np.random.RandomState):
            self.random_state_ = random_state
        elif random_state:
            try:
                self.random_state_ = np.random.RandomState(random_state)
            except TypeError as e:
                raise Exception("Invalid seed: " + str(e))
        else:
            self.random_state_ = np.random.mtrand._rand

        self.teacher_forcing = teacher_forcing
        self.silent = silent
        self.initweights()

    def initweights(self):
        # initialize recurrent weights:
        # begin with a random matrix centered around zero:
        W = self.random_state_.rand(self.n_reservoir, self.n_reservoir) - 0.5
        # delete the fraction of connections given by (self.sparsity):
        W[self.random_state_.rand(*W.shape) < self.sparsity] = 0
        # compute the spectral radius of these weights:
        radius = np.max(np.abs(np.linalg.eigvals(W)))
        # rescale them to reach the requested spectral radius:
        self.W = W * (self.spectral_radius / radius)

        # random input weights:
        self.W_in = self.random_state_.rand(
            self.n_reservoir, self.n_inputs) * 2 - 1
        # random feedback (teacher forcing) weights:
        self.W_feedb = self.random_state_.rand(
            self.n_reservoir, self.n_outputs) * 2 - 1

    def _update(self, state, input_pattern, output_pattern):
        """performs one update step.

        i.e., computes the next network state by applying the recurrent weights
        to the last state & and feeding in the current input and output patterns
        """
        if self.teacher_forcing:
            preactivation = (np.dot(self.W, state)
                             + np.dot(self.W_in, input_pattern)
                             + np.dot(self.W_feedb, output_pattern))

        else:
            preactivation = (np.dot(self.W, state)
                             + np.dot(self.W_in, input_pattern))

        return (np.tanh(preactivation)
                + self.noise * (self.random_state_.rand(self.n_reservoir) - 0.5))

    def _scale_inputs(self, inputs):
        """for each input dimension j: multiplies by the j'th entry in the
        input_scaling argument, then adds the j'th entry of the input_shift
        argument."""
        if self.input_scaling is not None:
            inputs = np.dot(inputs, np.diag(self.input_scaling))
        if self.input_shift is not None:
            inputs = inputs + self.input_shift
        return inputs

    def _scale_teacher(self, teacher):
        """multiplies the teacher/target signal by the teacher_scaling argument,
        then adds the teacher_shift argument to it."""
        if self.teacher_scaling is not None:
            teacher = teacher * self.teacher_scaling
        if self.teacher_shift is not None:
            teacher = teacher + self.teacher_shift
        return teacher

    def _unscale_teacher(self, teacher_scaled):
        """inverse operation of the _scale_teacher method."""
        if self.teacher_shift is not None:
            teacher_scaled = teacher_scaled - self.teacher_shift
        if self.teacher_scaling is not None:
            teacher_scaled = teacher_scaled / self.teacher_scaling
        return teacher_scaled

    def fit(self, inputs, outputs, inspect=False):
        # transform any vectors of shape (x,) into vectors of shape (x,1):
        if inputs.ndim < 2:
            inputs = np.reshape(inputs, (len(inputs), -1))
        if outputs.ndim < 2:
            outputs = np.reshape(outputs, (len(outputs), -1))
        # transform input and teacher signal:
        inputs_scaled = self._scale_inputs(inputs)
        teachers_scaled = self._scale_teacher(outputs)

        #if not self.silent:
        #    print("harvesting states...")
        # step the reservoir through the given input,output pairs:
        states = np.zeros((inputs.shape[0], self.n_reservoir))
        for n in range(1, inputs.shape[0]):
            states[n, :] = self._update(states[n - 1], inputs_scaled[n, :],
                                        teachers_scaled[n - 1, :])

        # learn the weights, i.e. find the linear combination of collected
        # network states that is closest to the target output
        #if not self.silent:
        #    print("fitting...")
        # we'll disregard the first few states:
        transient = min(int(inputs.shape[1] / 10), 100)
        # include the raw inputs:
        extended_states = np.hstack((states, inputs_scaled))
        # Solve for W_out:
        self.W_out = np.dot(np.linalg.pinv(extended_states[transient:, :]),
                            self.inverse_out_activation(teachers_scaled[transient:, :])).T

        # remember the last state for later:
        self.laststate = states[-1, :]
        self.lastinput = inputs[-1, :]
        self.lastoutput = teachers_scaled[-1, :]

        # optionally visualize the collected states
        if inspect:
            from matplotlib import pyplot as plt
            # (^-- we depend on matplotlib only if this option is used)
            plt.figure(
                figsize=(states.shape[0] * 0.0025, states.shape[1] * 0.01))
            plt.imshow(extended_states.T, aspect='auto',
                       interpolation='nearest')
            plt.colorbar()
        
        #if not self.silent:
        #    print("training error:")
        # apply learned weights to the collected states:
        pred_train = self._unscale_teacher(self.out_activation(
            np.dot(extended_states, self.W_out.T)))
        #if not self.silent:
        #    print(np.sqrt(np.mean((pred_train - outputs)**2)))
        return pred_train
    
    def predict(self, inputs, continuation=True):
        """
        Apply the learned weights to the network's reactions to new input.

        Args:
            inputs: array of dimensions (N_test_samples x n_inputs)
            continuation: if True, start the network from the last training state

        Returns:
            Array of output activations
        """
        if inputs.ndim < 2:
            inputs = np.reshape(inputs, (len(inputs), -1))
        n_samples = inputs.shape[0]

        if continuation:
            laststate = self.laststate
            lastinput = self.lastinput
            lastoutput = self.lastoutput
        else:
            laststate = np.zeros(self.n_reservoir)
            lastinput = np.zeros(self.n_inputs)
            lastoutput = np.zeros(self.n_outputs)

        inputs = np.vstack([lastinput, self._scale_inputs(inputs)])
        states = np.vstack(
            [laststate, np.zeros((n_samples, self.n_reservoir))])
        outputs = np.vstack(
            [lastoutput, np.zeros((n_samples, self.n_outputs))])

        for n in range(n_samples):
            states[
                n + 1, :] = self._update(states[n, :], inputs[n + 1, :], outputs[n, :])
            outputs[n + 1, :] = self.out_activation(np.dot(self.W_out,
                                                           np.concatenate([states[n + 1, :], inputs[n + 1, :]])))

        return self._unscale_teacher(self.out_activation(outputs[1:]))
    


