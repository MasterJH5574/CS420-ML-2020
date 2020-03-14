class LinearRegression2(object):
    def __init__(self):
        self.W = None

    def train(self, X, Y, display, learning_rate=1e-3, reg=1e-5, reg_type='L2', num_iters=2000,
              batch_size=128):
        num_train, feat_dim = X.shape
        num_classes = 10
        self.W = 0.001 * np.random.randn(feat_dim, num_classes).transpose()
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
            self.W -= learning_rate * grad

            if display and i % 100 == 0:
                print("In iteration {}/{} , the loss is {}".format(i, num_iters, loss))
        return loss_history

    def loss_grad(self, X, Y, reg):
        data_num, feat_dim = X.shape
        class_num = self.W.shape[0]

        loss = 0
        grad = np.zeros([class_num, feat_dim])
        for data in range(data_num):
            sum = 0

            exp_sum = np.exp(np.sum(X[data] * self.W, axis=1))
            sum = exp_sum.sum()
            loss += -np.log(exp_sum[Y[data]] / sum)

            for i in range(class_num):
                p = exp_sum[i] / sum
                if i == Y[data]:
                    p -= 1
                grad[i] += X[data] * p

        return loss / data_num, grad / data_num

    def l1_loss(self, X, Y, reg):
        # Todo 2
        loss, grad = self.loss_grad(X, Y, reg)

        loss = loss + reg * abs(self.W).sum()
        grad = grad + reg

        return loss, grad

    def l2_loss(self, X, Y, reg):
        # Todo 3
        loss, grad = self.loss_grad(X, Y, reg)

        loss = loss + reg * np.sum(self.W * self.W)
        grad = grad + 2 * reg * self.W

        return loss, grad

    def predict(self, X):
        # Todo 4
        data_num, feat_dim = X.shape
        class_num = self.W.shape[0]
        Y_pred = np.zeros(data_num)

        for data in range(data_num):
            max_p = 0
            pos = 0

            exp_sum = np.exp(np.sum(self.W * X[data], axis=1))
            Y_pred[data] = np.argmax(exp_sum)

        return Y_pred
