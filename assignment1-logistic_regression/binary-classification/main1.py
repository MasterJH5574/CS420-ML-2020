lr_param = 1.5e-6
reg_param = 0.01

model = LinearRegression1()
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