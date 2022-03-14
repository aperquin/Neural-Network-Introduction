import torch
import pandas

class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, filepath, input_cols, output_cols):
        df = pandas.read_csv(filepath)
        self.input_data = df[input_cols].values.tolist()
        self.output_data = df[output_cols].values.tolist()

    def __getitem__(self, idx):
        return torch.tensor(self.input_data[idx]), torch.tensor(self.output_data[idx])
        
    def __len__(self):
        return len(self.input_data)


# If we called this script instead of importing it, we test whether it functions correctly.
if __name__ == "__main__":
    dataset = DummyDataset("train_data.csv", ["x", "y"], ["x+y"])
    print(dataset[0:2])
    print(len(dataset))
