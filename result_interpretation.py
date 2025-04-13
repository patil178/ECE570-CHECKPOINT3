import pandas as pd
import matplotlib.pyplot as plt

# 1) Read CSV into a pandas DataFrame
df = pd.read_csv("results__full_1.csv")

# 2) Plot Training Loss vs. Epoch
plt.figure()  # separate figure
plt.plot(df["epoch"], df["train_loss"])
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss vs. Epoch")
plt.show()

# 3) Plot Validation Loss vs. Epoch
plt.figure()  # separate figure
plt.plot(df["epoch"], df["val_loss"])
plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.title("Validation Loss vs. Epoch")
plt.show()

# 4) Plot R² vs. Epoch
plt.figure()  # separate figure
plt.plot(df["epoch"], df["r2"])
plt.xlabel("Epoch")
plt.ylabel("R²")
plt.title("R² vs. Epoch")
plt.show()

# 5) Plot RMSE vs. Epoch
plt.figure()  # separate figure
plt.plot(df["epoch"], df["rmse"])
plt.xlabel("Epoch")
plt.ylabel("RMSE")
plt.title("RMSE vs. Epoch")
plt.show()
