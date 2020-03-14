def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class LinearRegression1(object):
    def __init__(self):
        self.W = None

    def train(self, X, Y, display, learning_rate=1e-3, reg=1e-5, reg_type='L2', num_iters=2000, batch_size=128):
        num_train, feat_dim = X.shape
        self.W = 0.001 * np.random.randn(feat_dim)
        loss_history = []
        for i in range(num_iters):
            batch_indices = np.random.choice(num_train, batch_size, replace=True)
            X_batch = X[batch_indices]
            Y_batch = Y[batch_indices]
            if reg_type == 'L1':
                loss, grad = self.l1_loss(X_batch, Y_batch, reg)
            else:
                loss, grad = self.l2_loss(X_batch, Y_batch, reg)
            loss_history.append(loss)

            # Todo 1
            # lr = learning_rate / (i // 800 + 1)
            # self.W -= lr * grad
            self.W -= learning_rate * grad

            if display and i % 100 == 0:
                print("In iteration {}/{} , the loss is {}".format(i, num_iters, loss))
        return loss_history

    def loss_grad(self, X, Y, reg):
        num, feat_dim = X.shape

        loss = 0
        grad = np.zeros(feat_dim)
        for i in range(num):
            z = sigmoid(np.sum(self.W * X[i]))
            loss += -Y[i] * math.log(z) - (1 - Y[i]) * math.log(1 - z)
            grad += (z - Y[i]) * X[i]

        return loss / num, grad / num

    def l1_loss(self, X, Y, reg):
        # Todo 2
        loss, grad = self.loss_grad(X, Y, reg)

        loss = loss + reg * np.sum(abs(self.W))
        grad = grad + reg

        return loss, grad

    def l2_loss(self, X, Y, reg):
        # Todo 3
        loss, grad = self.loss_grad(X, Y, reg)

        loss = loss + reg * np.sum(self.W * self.W)
        grad = grad + 2 * reg * self.W

        return loss, grad

    def predict(self, X, threshold=0.5):
        # Todo 4
        num, feat_dim = X.shape
        Y_pred = np.zeros(num)

        for i in range(num):
            val = sigmoid(np.sum(self.W * X[i]))
            if val >= threshold:
                Y_pred[i] = 1
            else:
                Y_pred[i] = 0

        return Y_pred
