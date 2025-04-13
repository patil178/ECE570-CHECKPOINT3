import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


df = pd.read_excel("au2c00122_si_002.xlsx", nrows=1085)
print("Data shape:", df.shape)


X = df.iloc[:, :-2].values
y = df.iloc[:, -2].values
print("Features shape:", X.shape)
print("Labels shape:", y.shape)

# Convert to tensors
X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y, dtype=torch.float32).to(device)


class FCNN(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 128, 16], output_dim=1):
        super(FCNN, self).__init__()
        layers = []
        in_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.PReLU())
            in_dim = h_dim
        
        layers.append(nn.Linear(in_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

model = FCNN(input_dim=X.shape[1], hidden_dims=[512, 128, 16], output_dim=1)
model = model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=0.01)


num_epochs = 20000

epoch_list      = []
train_loss_list = []
val_loss_list   = []
r2_list         = []
rmse_list       = []

for epoch in range(num_epochs):
    indices = np.random.permutation(len(X_tensor))
    train_size = int(0.8 * len(X_tensor))
    train_idx = indices[:train_size]
    val_idx   = indices[train_size:]
    
    X_train = X_tensor[train_idx]
    y_train = y_tensor[train_idx].view(-1, 1)  
    
    X_val   = X_tensor[val_idx]
    y_val   = y_tensor[val_idx].view(-1, 1)    
    

    model.train()
    y_pred_train = model(X_train)
    loss_train = criterion(y_pred_train, y_train)
    
    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()
    

    model.eval()
    with torch.no_grad():
        y_pred_val = model(X_val)
        loss_val = criterion(y_pred_val, y_val)
    

    if (epoch + 1) % 500 == 0:
        # R^2 = 1 - (SSres / SStot)
        ss_res = torch.sum((y_val - y_pred_val) ** 2)
        ss_tot = torch.sum((y_val - torch.mean(y_val)) ** 2)
        r2 = 1 - ss_res / ss_tot
        
        rmse = torch.sqrt(loss_val)
        mae = torch.mean(torch.abs(y_pred_val - y_val))
        
        epoch_list.append(epoch + 1)
        train_loss_list.append(loss_train.item())
        val_loss_list.append(loss_val.item())
        r2_list.append(r2.item())
        rmse_list.append(rmse.item())
        
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Train Loss: {loss_train.item():.4f}")
        print(f"  Val   Loss: {loss_val.item():.4f}")
        print(f"  R^2:       {r2.item():.4f}")
        print(f"  RMSE:      {rmse.item():.4f}")
        print(f"  MAE:       {mae.item():.4f}")

print("Training complete!")


results_df = pd.DataFrame({
    'epoch': epoch_list,
    'train_loss': train_loss_list,
    'val_loss': val_loss_list,
    'r2': r2_list,
    'rmse': rmse_list
})
results_df.to_csv("results__full_1.csv", index=False)
print("Results saved to results__full.csv")


df_test = pd.read_excel("au2c00122_si_002_test.xlsx")
print("Test data shape:", df_test.shape)

X_test = df_test.iloc[:, :-2].values
y_test = df_test.iloc[:, -2].values

X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

model.eval()
with torch.no_grad():
    y_pred_test = model(X_test_tensor)
    
    test_loss = criterion(y_pred_test, y_test_tensor)
    
    ss_res_test = torch.sum((y_test_tensor - y_pred_test) ** 2)
    ss_tot_test = torch.sum((y_test_tensor - torch.mean(y_test_tensor)) ** 2)
    r2_test = 1 - ss_res_test / ss_tot_test
    
    test_rmse = torch.sqrt(test_loss)
    test_mae = torch.mean(torch.abs(y_pred_test - y_test_tensor))
    
    print("\n=== Test Set Results ===")
    print(f"Test Loss (MSE): {test_loss.item():.4f}")
    print(f"Test R^2:        {r2_test.item():.4f}")
    print(f"Test RMSE:       {test_rmse.item():.4f}")
    print(f"Test MAE:        {test_mae.item():.4f}")
