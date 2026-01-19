w = 0.0
b = 0.0
x = 1
y = 7
lr = 0.1

for step in range(10000):
    y_pred = w * x + b
    error = y_pred - y

    w = w - lr * error * x
    b = b - lr * error

    print(step, w, b)

print(w*x+b)
