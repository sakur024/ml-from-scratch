import numpy as np

# Dataset
X = np.array([1, 2, 3, 4], dtype=float)
Y = np.array([5, 8, 11, 14], dtype=float)

# Parameters
w = 0.0
b = 0.0

lr = 0.1
epochs = 50


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


for epoch in range(epochs):
    # 1. Predict for all points
    y_pred = w * X + b

    # 2. Compute error for all points
    error = y_pred - Y

    # 3. Gradients (averaged over all points)
    dw = np.mean(error * X)
    db = np.mean(error)

    # 4. Update parameters
    w = w - lr * dw
    b = b - lr * db

    # 5. Loss
    loss = mse(Y, y_pred)

    print(
        f"Epoch {epoch:02d} | "
        f"Loss: {loss:.4f} | "
        f"w: {w:.3f}, b: {b:.3f}"
    )
