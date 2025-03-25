import numpy as np


class ReplayBuffer:
    def __init__(self, max_len, batch_size):
        self.rng = np.random.default_rng()
        self.max_len, self.batch_size = max_len, batch_size
        self.data = []

    def __len__(self):
        return len(self.data)

    def add(self, _sample):
        sample = np.concatenate([np.asarray(item).flatten() for item in _sample])
        if len(self.data) == 0:
            self.data = sample[None, :]
        elif self.data.shape[0] < self.max_len:
            self.data = np.vstack((self.data, sample[None, :]))
        else:
            self.data = np.vstack((self.data[1:, :], sample[None, :]))
        return

    def sample(self, sample_index=None):
        """
        Randomly sample batch_size data from buffer
        """
        if len(self.data) < self.batch_size:
            raise Exception('Not enough buffer data for sampling.')
        else:
            if sample_index is None:
                idx = self.rng.choice(self.data.shape[0], self.batch_size, replace=False)
                return self.data[idx, :]
            else:
                return self.data[sample_index]

    def load_from_npy(self, file_path):
        data = np.load(file_path)
        self.data = data
