import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist['data'], mnist['target'].astype(np.int32)

X = X / 255.0
y_encoded = np.eye(10)[y]

rn = np.random.permutation(len(X))
X, y_encoded, y = X[rn], y_encoded[rn], y[rn]

X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y_encoded[:60000], y_encoded[60000:]
y_test_labels = y[60000:]

def relu(z):
    return np.maximum(0, z)

def relu_deriv(z):
    return (z > 0).astype(float)

def softmax(z):
    return np.exp(z - np.max(z)) / np.sum(np.exp(z - np.max(z)))


#learning on a training set that contais 60000 fotos
def train_nn(X, y, alpha=0.01, epochs=10):
    np.random.seed(1)
    w1 = np.random.randn(784, 64) * 0.01
    b1 = np.zeros(64)
    w2 = np.random.randn(64, 64) * 0.01
    b2 = np.zeros(64)
    w3 = np.random.randn(64, 10) * 0.01
    b3 = np.zeros(10)
# random initialisation for better results
    loss_history = []
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        for i in range(len(X)):
            x = X[i]
            y_true = y[i]

            z1 = np.dot(x, w1) + b1
            a1 = relu(z1)
            z2 = np.dot(a1, w2) + b2
            a2 = relu(z2)
            z3 = np.dot(a2, w3) + b3
            y_pred = softmax(z3)

            if np.argmax(y_pred) == np.argmax(y_true):
                correct += 1

            # Loss
            loss = -np.sum(y_true * np.log(y_pred + 1e-8))
            total_loss += loss

            # Backward propagation
            err = y_pred - y_true
            dw3 = np.outer(a2, err)
            db3 = err

            d2 = np.dot(err, w3.T) * relu_deriv(z2)
            dw2 = np.outer(a1, d2)
            db2 = d2

            delta1 = np.dot(d2, w2.T) * relu_deriv(z1)
            dw1 = np.outer(x, delta1)
            db1 = delta1

            # recount all parameters
            w3 -= alpha * dw3
            b3 -= alpha * db3

            w2 -= alpha * dw2
            b2 -= alpha * db2

            w1 -= alpha * dw1
            b1 -= alpha * db1

        avg_loss = total_loss / len(X)
        acc = correct / len(X) * 100
        print(f"Epoch {epoch+1}   Loss: {avg_loss:.4f}  Accuracy: {acc:.2f}%")
        loss_history.append(avg_loss)

    return w1, b1, w2, b2, w3, b3, loss_history

def test(x, w1, b1, w2, b2, w3, b3):
    z1 = np.dot(x, w1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, w2) + b2
    a2 = relu(z2)
    z3 = np.dot(a2, w3) + b3
    return softmax(z3)

w1, b1, w2, b2, w3, b3, loss_history = train_nn(X_train, y_train, alpha=0.01, epochs=2)

plt.figure(figsize=(8, 5))
plt.plot(loss_history, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Convergence over epochs")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("loss_plot.pdf")
plt.show()

correct = 0
for i in range(len(X_test)):
    y_pred = test(X_test[i], w1, b1, w2, b2, w3, b3)
    if np.argmax(y_pred) == y_test_labels[i]:
        correct += 1
print(f"Test accuracy: {correct / len(X_test) * 100:.2f}%")

for i in range(3):
    img = X_test[i].reshape(28, 28)
    pred = np.argmax(test(X_test[i], w1, b1, w2, b2, w3, b3))
    true = y_test_labels[i]
    plt.imshow(img, cmap='gray')
    plt.title(f"Predicted: {pred} | True: {true}")
    plt.axis('off')
    plt.show()