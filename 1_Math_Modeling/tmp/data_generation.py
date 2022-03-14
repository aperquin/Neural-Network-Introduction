import os
import numpy as np
import pandas as pd

def generate_data(output_file, nb_samples):
    x = np.random.randint(1, 10+1, nb_samples) * np.random.rand(nb_samples)
    y = np.random.randint(1, 10+1, nb_samples) * np.random.rand(nb_samples)

    df = pd.DataFrame()
    df["x"] = x
    df["y"] = y
    df["x+y"] = x + y
    df["x-y"] = x - y
    df["x*y"] = x * y
    df["x^2"] = x * x
    df["log(x)"] = np.log(x)

    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    generate_data("train_data.csv", 1000)
    generate_data("validation_data.csv", 100)
    generate_data("evaluation_data.csv", 100)