from __future__ import annotations


class DataLoader:  # minimal placeholder
    def __init__(self, data, batch_size=1, shuffle=False):  # type: ignore[no-untyped-def]
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        return iter(self.data)
