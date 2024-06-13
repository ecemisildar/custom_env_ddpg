import numpy as np

class ReplayBuffer:
    def __init__(self, max_size, sequence_length, input_shape, n_actions):
        self.mem_size = max_size
        self.seq_l = sequence_length
        self.dim_input = input_shape
        self.dim_action = n_actions
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float16)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float16)
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=np.float16)
        self.reward_memory = np.zeros(self.mem_size,)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):

        unbalance_p = True
        max_mem = min(self.mem_cntr, self.mem_size) - self.seq_l

        p_indices = None

        if unbalance_p:
            p_indices = np.arange(1, max_mem + 1)*0.5
            p_indices = p_indices / np.sum(p_indices)

        # print(max_mem)
        batch = np.random.choice(max_mem, batch_size, replace=False, p=p_indices)

        state_seqs = np.zeros((batch_size, self.seq_l, *self.dim_input))
        states2_seqs = np.zeros((batch_size, self.seq_l, *self.dim_input))
        action_seqs = np.zeros((batch_size, self.seq_l, self.dim_action))
        reward_seqs = np.zeros((batch_size, self.seq_l, 1))
        done_seqs = np.zeros((batch_size, self.seq_l, 1), dtype=bool)

        print(f"state sequence shape: {state_seqs.shape}")
        print(f"action sequence shape: {action_seqs.shape}")
        
        for i, b in enumerate(batch):
            state_seqs[i,:,:] = self.state_memory[b:b+self.seq_l]
            states2_seqs[i,:,:] = self.new_state_memory[b:b+self.seq_l]
            action_seqs[i,:,:] = self.action_memory[b:b+self.seq_l]
            reward_seqs[i,:,:] = self.reward_memory[b:b+self.seq_l].reshape(-1,1)
            done_seqs[i,:,:] = self.terminal_memory[b:b+self.seq_l].reshape(-1,1)

        
        
        return state_seqs, action_seqs, reward_seqs, states2_seqs, done_seqs