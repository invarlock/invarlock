from __future__ import annotations


class DataLoader:  # minimal placeholder
    def __init__(
        self,
        data: object,
        batch_size: int = 1,
        shuffle: bool = False,
    ) -> None:
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        return iter(self.data)
