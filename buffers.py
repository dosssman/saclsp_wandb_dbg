import numpy as np

# Default buffer for SAC LSP
class SimpleReplayBuffer( object):
    def __init__( self, obs_dim, action_dim, max_size = 10000, batch_size = 128):
        self.max_size = max_size
        self.batch_size = batch_size
        self.current_size = 0
        self.current_index = 0
        self.seed = 42

        self.Do = obs_dim
        self.Da = action_dim
        # TODO: Add support for disc. action space tasks too !

        self.observations = np.zeros([self.max_size, self.Do])
        self.actions = np.zeros( [self.max_size, self.Da])
        self.rewards = np.zeros( self.max_size)
        self.terminals = np.zeros( self.max_size)
        self.next_observations = np.zeros( [self.max_size, self.Do])

    def add_transition( self, observation, action, reward, next_observation, terminal):
        self.observations[self.current_index] = np.reshape( observation, self.Do)
        self.actions[self.current_index] = action
        self.rewards[self.current_index] = reward
        self.terminals[self.current_index] = terminal
        self.next_observations[self.current_index] = np.reshape( next_observation, self.Do)

        self.current_index += 1
        self.current_index %= self.max_size
        self.current_size = max( self.current_index, self.current_size)

    def set_seed( self, seed = None):
        if seed is not None:
            # TODO: Numeric control on seed value
            self.seed = seed

        np.random.seed( self.seed)

    @property
    def size(self):
        return self.current_size

    @property
    def is_full(self):
        return self.current_index == self.max_size - 1

    def is_ready_for_sample( self, batch_size=None):
        if batch_size is None:
            return self.size >= self.batch_size

        return self.size >= batch_size

    def sample( self, batch_size = None):
        sample_batch_size = self.batch_size if batch_size is None else batch_size
        assert self.is_ready_for_sample(sample_batch_size), 'Not enough data to sample'

        batch_indices = np.random.randint(0, self.size, size=sample_batch_size)

        return self.observations[batch_indices], \
                self.actions[batch_indices], \
                self.rewards[batch_indices], \
                self.next_observations[batch_indices], \
                self.terminals[batch_indices]
