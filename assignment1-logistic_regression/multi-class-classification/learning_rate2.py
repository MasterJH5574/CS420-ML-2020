reg = 0.01
reg_types = ['L1', 'L2']
L1_loss = []
L2_loss = []
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2)
L1_lr_val_acc = []
L2_lr_val_acc = []

# Todo 5:
learning_rates = [7e-5, 4e-5, 1e-5, 7e-6, 4e-6, 1e-6, 7e-7]
for type in range(2):
    for i in range(len(learning_rates)):
        model = LinearRegression2()
        learning_rate = learning_rates[i]
        loss_history = model.train(X_train, Y_train, False, learning_rate, reg, reg_types[type])
        accuracy = np.mean(model.predict(X_val) == Y_val)
        if type == 0:
            L1_loss.append(loss_history)
            L1_lr_val_acc.append(accuracy)
        else:
            L2_loss.append(loss_history)
            L2_lr_val_acc.append(accuracy)
        print(reg_types[type], learning_rate, accuracy)

# visulize the relationship between lr and loss
for i, lr in enumerate(learning_rates):
    L1_loss_label = str(lr) + 'L1'
    L2_loss_label = str(lr) + 'L2'
    L1_loss_i = L1_loss[i]
    L2_loss_i = L2_loss[i]
    ave_L1_loss = np.zeros_like(L1_loss_i)
    ave_L2_loss = np.zeros_like(L2_loss_i)
    ave_step = 20
    for j in range(len(L1_loss_i)):
        if j < ave_step:
            ave_L1_loss[j] = np.mean(L1_loss_i[0: j + 1])
            ave_L2_loss[j] = np.mean(L2_loss_i[0: j + 1])
        else:
            ave_L1_loss[j] = np.mean(L1_loss_i[j - ave_step + 1: j + 1])
            ave_L2_loss[j] = np.mean(L2_loss_i[j - ave_step + 1: j + 1])
    x = range(len(L1_loss_i))
    plt.plot(x, ave_L1_loss, label=L1_loss_label)
    plt.plot(x, ave_L2_loss, label=L2_loss_label)

plt.legend()
plt.xlabel('high-parameter lr')
plt.ylabel('Loss')
plt.show()

# visulize the relationship between lr and accuracy
x = range(len(learning_rates))
plt.plot(x, L1_lr_val_acc, label='L1_val_acc')
plt.plot(x, L2_lr_val_acc, label='L2_val_acc')
plt.xticks(x, learning_rates)
plt.margins(0.08)
plt.legend()
plt.xlabel('high-parameter lr')
plt.ylabel('Validation Accuracy')
plt.show()