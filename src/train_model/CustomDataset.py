import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, input: list[list[float]], output: list[int], model_input_neuron: int):
        if len(input) != len(output):
            raise ValueError("Input and output data must have the same length")

        self.input: list[list[float]] = input
        self.output: list[int] = output
        self.model_input_neuron = model_input_neuron
        # print(len(input), len(output))

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        input_data: list[float] = self.input[idx]
        # print(input_data)
        while len(input_data) < self.model_input_neuron:
            input_data.append(0)
        input_tensor = torch.tensor(input_data, dtype=torch.float32)
        label_tensor = torch.tensor(self.output[idx], dtype=torch.long)
        return input_tensor, label_tensor
