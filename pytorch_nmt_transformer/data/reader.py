from pathlib import Path
import numpy as np
from torch.utils.data import Dataset


class DataReader(Dataset):
    def __init__(self, dataset_path: str) -> None:
        self.path = Path(dataset_path)
        if not self.path.is_file():
            raise FileNotFoundError(
                f"Given path does not exists or not a file: {str(self.path)}"
            )

        self.text_lines = self.path.read_text(encoding="utf-8").splitlines()
        self.text_lines = [line.split("\t") for line in self.text_lines]

    def __len__(self):
        return len(self.text_lines)

    def __getitem__(self, index):
        context = self.text_lines[index][1]
        target = self.text_lines[index][0]
        return context, target
