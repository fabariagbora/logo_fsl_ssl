import random
from torch.utils.data import Sampler

class FewShotEpisodeSampler(Sampler):
    def __init__(self, dataset, n_way, k_shot, m_query, num_episodes):
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.m_query = m_query
        self.num_episodes = num_episodes

        # Build index for each class
        self.index_per_class = {}
        for cls in self.dataset.classes:
            indices = [i for i, (_, label) in enumerate(self.dataset.samples) if label == cls]
            if len(indices) >= (self.k_shot + self.m_query):
                self.index_per_class[cls] = indices

        self.classes = list(self.index_per_class.keys())
        assert len(self.classes) >= self.n_way, (
            f"Not enough valid classes with at least {self.k_shot + self.m_query} images! "
            f"Available: {len(self.classes)}, Required per episode: {self.n_way}"
        )

        print(f"FewShotEpisodeSampler: Using {len(self.classes)} classes with >= {self.k_shot + self.m_query} images each.")

    def __iter__(self):
        for _ in range(self.num_episodes):
            batch = []
            selected_classes = random.sample(self.classes, self.n_way)
            for cls in selected_classes:
                idxs = random.sample(self.index_per_class[cls], self.k_shot + self.m_query)
                batch.extend(idxs)
            yield batch

    def __len__(self):
        return self.num_episodes
