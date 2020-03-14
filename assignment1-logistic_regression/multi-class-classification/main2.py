lr_param = 7e-6
reg_param = 0.01
model = LinearRegression2()
loss_history = model.train(X_train, Y_train, True, lr_param, reg_param, 'L2')
pred = model.predict(X_test)
acc = np.mean(pred == Y_test)
print("The Accuracy is {}\n".format(acc))
x = range(len(loss_history))
plt.plot(x, loss_history, label='Loss')
plt.legend()
plt.xlabel('Iteration Num')
plt.ylabel('Loss')
plt.show()

W = model.W
for digit in range(10):
    w = np.reshape(np.delete(W[digit], -1), (28, -1))
    w_min = np.min(w)
    w_max = np.max(w)
    w = (w - w_min) / (w_max - w_min) * 255.0
    print(digit)
    plt.imshow(w, cmap=plt.cm.gray)
    # plt.imshow(w)
    plt.axis('off')
    plt.show()
