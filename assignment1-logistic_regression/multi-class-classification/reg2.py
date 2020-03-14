learning_rate = 7e-6
reg_types = ['L1', 'L2']
L1_reg_val_acc = []
L2_reg_val_acc = []
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2)
best_L1_model = None
best_L2_model = None
best_L1_reg = 0
best_L2_reg = 0

# Todo 6:
regs = [0.001, 0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040]
for type in range(2):
    best_accuracy = 0
    for i in range(len(regs)):
        model = LinearRegression2()
        reg = regs[i]
        loss_history = model.train(X_train, Y_train, False, learning_rate, reg, reg_types[type])
        accuracy = np.mean(model.predict(X_val) == Y_val)
        if type == 0:
            L1_reg_val_acc.append(accuracy)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_L1_reg = reg
                best_L1_model = model
        else:
            L2_reg_val_acc.append(accuracy)
            if (accuracy > best_accuracy):
                best_accuracy = accuracy
                best_L2_reg = reg
                best_L2_model = model
        print(reg_types[type], reg, accuracy)

#visuliza the relation of regularization parameter and validation accuracy
x = range(len(regs))
plt.plot(x, L1_reg_val_acc, label='L1_val_acc')
plt.plot(x, L2_reg_val_acc, label='L2_val_acc')
plt.xticks(x, regs)
plt.margins(0.08)
plt.legend()
plt.xlabel('high-parameter reg')
plt.ylabel('Validation Accuracy')
plt.show()

#Compute the performance of best model on the test set
L1_pred = best_L1_model.predict(X_test)
L1_acc = np.mean(L1_pred == Y_test)
print("The Accuracy with L1 regularization parameter {} is {}\n".format(best_L1_reg, L1_acc))
L2_pred = best_L2_model.predict(X_test)
L2_acc = np.mean(L2_pred == Y_test)
print("The Accuracy with L1 regularization parameter {} is {}\n".format(best_L2_reg, L2_acc))