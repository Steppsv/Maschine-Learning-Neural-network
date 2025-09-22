import csv
import numpy as np
import matplotlib.pyplot as plt

data = []

with open("Real_estate.csv", newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for row in reader:
        try:
            numeric_row = [float(item) for item in row]
            data.append(numeric_row)
        except ValueError:
            continue
data = np.array(data)

X2 = data[:,2]   #x train set
X4 = data[:,4]   #y Train set
Y  = data[:,7]   #Y

def bgd(x1, x2, y, alp, maxit): #Defining function BGD
    teta0 = 0.0
    teta1 = 0.0
    teta2 = 0.0
    m = len(y)
    loss = []
    for i in range(maxit):
        h_0 = teta0 + teta1 * x1 + teta2 * x2
        dh = h_0 - y

        dteta0 = (1 / m) * np.sum(dh)
        dteta1 = (1 / m) * np.sum(dh * x1)
        dteta2 = (1 / m) * np.sum(dh * x2)

        teta0 -= alp * dteta0 #searching for perfect teta
        teta1 -= alp * dteta1
        teta2 -= alp * dteta2

        mse = (1 / (2 * m)) * np.sum(dh ** 2) #MSE counting and saving to an array
        loss.append(mse)
        tetas = [teta0, teta1, teta2]
    return tetas, loss

def sgd(x1, x2, y, alp, epoch):
    teta0 = 0.0
    teta1 = 0.0
    teta2 = 0.0
    m = len(y)
    loss = []
    for _ in range(epoch): # in  SGD loop of epoches
        total_loss = 0
        for i in range(m): # anylising all date for each ep
            h = teta0 + teta1 * x1[i] + teta2 * x2[i]
            d_h = h - y[i]

            teta0 -= alp * d_h #update
            teta1 -= alp * d_h * x1[i]
            teta2 -= alp * d_h * x2[i]

            total_loss += (d_h ** 2) #summ of errors
        loss.append((1 / (2 * m)) * total_loss)
    tetas = [teta0, teta1, teta2]
    return tetas, loss

def test_mse(teta,x1,x2,y):
    total_loss = 0
    for i in range(len(x1)):
        h = teta[0] + teta[1] * x1[i] + teta[2] * x2[i]
        d_h = h - y[i]
        total_loss += (d_h ** 2)

    return total_loss


sgd_X = np.arange(0,1000,10)
t_b, loss_bgd = bgd(X2[:int(0.8 * len(X2))], X4[:int(0.8 * len(X2))], Y[:int(0.8 * len(X2))], alp=0.0001, maxit=1000) #calling function
t_s, loss_sgd = sgd(X2[:int(0.8 * len(X2))], X4[:int(0.8 * len(X2))], Y[:int(0.8 * len(X2))], alp=0.0001, epoch=100)

h_bgd = t_b[0] + t_b[1] * X2 + t_b[2] * X4 #test examples to test MSE
h_sgd = t_s[0] + t_s[1] * X2 + t_s[2] * X4
mse_bgd = (1 / (2 * len( Y[:int(0.8 * len(X2))]))) * np.sum((h_bgd[:int(0.8 * len(X2))] - Y[:int(0.8 * len(X2))]) ** 2)
mse_sgd = (1 / (2 * len( Y[:int(0.8 * len(X2))]))) * np.sum((h_sgd[:int(0.8 * len(X2))] - Y[:int(0.8 * len(X2))]) ** 2)
mse_t_b_test = (1 / (2 * len( Y[int(0.8 * len(X2)):]))) * np.sum((h_bgd[int(0.8 * len(X2)):] - Y[int(0.8 * len(X2)):]) ** 2)
mse_t_s_test = (1 / (2 * len( Y[int(0.8 * len(X2)):]))) * np.sum((h_bgd[int(0.8 * len(X2)):] - Y[int(0.8 * len(X2)):]) ** 2)

print(f"MSE of a BGD : {mse_bgd:.2f}")
print(f"MSE of a SGD : {mse_sgd:.2f}")
print("LR = 0.001")
print("A total loss of a BGD was ",mse_t_b_test)
print("A total loss of a SGD was ",mse_t_s_test)
plt.plot(loss_bgd, label="BGD", linewidth=2)
plt.plot(sgd_X,loss_sgd, label="SGD", linewidth=3)
plt.xlabel("Iteration / Epoch")
plt.ylabel("MSE")
plt.title("Loss/Iterations: BGD-B SGD-Y 0.0001")
plt.legend()
plt.grid(True)
plt.show()


                            # Same for 0.001
t_b, loss_bgd = bgd(X2[:int(0.8 * len(X2))], X4[:int(0.8 * len(X2))], Y[:int(0.8 * len(X2))], alp=0.001, maxit=1000) #calling function
t_s, loss_sgd = sgd(X2[:int(0.8 * len(X2))], X4[:int(0.8 * len(X2))], Y[:int(0.8 * len(X2))], alp=0.001, epoch=100)
h_bgd = t_b[0] + t_b[1] * X2 + t_b[2] * X4 #test examples to test MSE
h_sgd = t_s[0] + t_s[1] * X2 + t_s[2] * X4
mse_bgd = (1 / (2 * len( Y[:int(0.8 * len(X2))]))) * np.sum((h_bgd[:int(0.8 * len(X2))] - Y[:int(0.8 * len(X2))]) ** 2)
mse_sgd = (1 / (2 * len( Y[:int(0.8 * len(X2))]))) * np.sum((h_sgd[:int(0.8 * len(X2))] - Y[:int(0.8 * len(X2))]) ** 2)
mse_t_b_test = (1 / (2 * len( Y[int(0.8 * len(X2)):]))) * np.sum((h_bgd[int(0.8 * len(X2)):] - Y[int(0.8 * len(X2)):]) ** 2)
mse_t_s_test = (1 / (2 * len( Y[int(0.8 * len(X2)):]))) * np.sum((h_bgd[int(0.8 * len(X2)):] - Y[int(0.8 * len(X2)):]) ** 2)

print(f"MSE of a BGD : {mse_bgd:.2f}")
print(f"MSE of a SGD : {mse_sgd:.2f}")
print("LR = 0.001")
print("A total loss of a BGD was ",mse_t_b_test)
print("A total loss of a SGD was ",mse_t_s_test)
plt.plot(loss_bgd, label="BGD", linewidth=2)
plt.plot(sgd_X,loss_sgd, label="SGD", linewidth=3)
plt.xlabel("Iteration / Epoch")
plt.ylabel("MSE")
plt.title("Loss/Iterations: BGD-B SGD-Y 0.001")
plt.legend()
plt.grid(True)
plt.show()



x2_range = np.linspace(X2.min(), X2.max(), 50)
x4_range = np.linspace(X4.min(), X4.max(), 50)
x2_grid, x4_grid = np.meshgrid(x2_range, x4_range)
y_pred_plane = t_b[0] + t_b[1] * x2_grid + t_b[2] * x4_grid  #3D Plot

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X2,X4,Y,label= "Parametric Curve")
ax.plot_surface(x2_grid,x4_grid,y_pred_plane,color='red', alpha=0.5)
plt.show()

#as I understood, unseen dataset was the 80/20 dividing of a dataset to achieve train and test set
