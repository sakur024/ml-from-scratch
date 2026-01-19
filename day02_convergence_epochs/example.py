# Day 2: Understanding epochs, convergence, and learning rate behavior

# Initial parameters
w = 0.0
b = 0.0

# Learning setup
lr = 0.1        # try 0.1, then 1.0, then 0.01
epochs = 50

# Simple dataset (y = 3x + 2)
X = [1, 2, 3, 4]
Y = [5, 8, 11, 14]


def mse(y_true, y_pred):
    return (y_true - y_pred) ** 2


for epoch in range(epochs):
    total_loss = 0

    for x, y in zip(X, Y):
        # prediction
        y_pred = w * x + b

        # error
        error = y_pred - y

        # parameter updates
        w = w - lr * error * x
        b = b - lr * error

        # accumulate loss
        total_loss += mse(y, y_pred)

    avg_loss = total_loss / len(X)

    print(
        f"Epoch {epoch:02d} | "
        f"Loss: {avg_loss:.4f} | "
        f"w: {w:.3f}, b: {b:.3f}"
    )
