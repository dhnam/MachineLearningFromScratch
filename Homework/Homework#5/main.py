import numpy as np
import csv


# Part 1 : Classes for ML Model

class MSELoss:
    def __init__(self):
        self._diff = None

    def __call__(self, output, y):
        return self.forward(output, y)
    
    def forward(self, output, y):
        self._diff = output - y
        square = np.square(self._diff)
        return np.mean(square) # (diff)^2 / (output_shape)

    def backward(self):
        # d(Loss) / d(diff) = 2 * diff / (output_shape) 
        return 2 * self._diff / np.prod(self._diff.shape)


class DenseLayer:
    def __init__(self, input_size, output_size, lr=0.001, mean=0, std=0.003):
        self.weight = np.random.normal(mean, std, [input_size, output_size])
        self.bias = np.zeros([output_size])
        self.lr = lr

    def forward(self, x):
        output = np.matmul(x, self.weight)
        output += self.bias
        return output
    
    def __call__(self, x):
        return self.forward(x)

    def backward(self, loss_grad, x):
        g_output_w = x.transpose()
        
        grad_w = np.matmul(g_output_w, loss_grad) # (feature_num, N) * (N, 1)
        grad_b = np.sum(loss_grad, axis=0)

        self.weight -= self.lr * grad_w
        self.bias -= self.lr * grad_b

class Network:
    # 'Baked' network
    def __init__(self):
        self.layer = DenseLayer(10, 1)
        self.loss = MSELoss()
        self._output = None
        self._x_prev = None

    def forward(self, x):
        return self.layer(x)

    def __call__(self, x):
        self._x_prev = x
        self._output = self.forward(x)
        return self._output

    def backprop(self, y):
        loss = self.loss(self._output, y)
        grad = self.loss.backward()
        self.layer.backward(grad, self._x_prev)
        return loss

class Trainer:
    def __init__(self, network, epoch):
        self.network = network
        self.epoch = epoch

    def train(self, x, y, print_freq=10):
        for i in range(self.epoch):
            output = self.network(x)
            loss = self.network.backprop(y)

            if i % print_freq == 0:
                print("Epoch :", i, "Loss :", loss, "Accuracy : %.4f" % self.eval_acc(output, y))

    def train_generator(self, generator_fn, args, print_freq=10):
        # train for mini batch generator
        for i in range(self.epoch):
            generator = generator_fn(*args)
            for x, y in generator:
                output = self.network(x)
                loss = self.network.backprop(y)

            if i % print_freq == 0:
                print("Epoch :", i, "Loss :", loss, "Accuracy : %.4f" % self.eval_acc(output, y))
    
    def eval_acc(self, output, y):
        mdiff = np.mean(np.abs((output - y) / y))
        return 1 - mdiff

    def test(self, x, y):
        output = self.network(x)
        loss = self.network.loss(output, y)
        acc = self.eval_acc(output, y)
        print("Loss :", loss, "Accuracy : %.4f" % self.eval_acc(output, y))
        return loss, acc

    
 # Part 2 : Helper functions

def load_abalone_dataset():
    with open('./data.csv') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader, None)
        rows = []
        for row in csvreader:
            rows.append(row)

    input_cnt, output_cnt = 10, 1
    data = np.zeros([len(rows), input_cnt+output_cnt])

    for n, row in enumerate(rows):
        if row[0] == 'M':
            data[n, 0] = 1
        elif row[0] == 'F':
            data[n, 1] = 1
        else:
            data[n, 2] = 1
        data[n, 3:] = row[1:]
    return data

def shuffle_data(data):
    np.random.shuffle(data)

def train_test_split(data, train_ratio=0.7):
    count = int(len(data) * train_ratio)
    x_train = data[:count, :-1]
    y_train = data[:count, -1:]
    x_test = data[count:, :-1]
    y_test = data[count:, -1:]
    return x_train, y_train, x_test, y_test

def minibatch(x_data, y_data, batch_size=10):
    if len(x_data) != len(y_data):
        return
    while x_data.tolist():
        x_batch, x_data = x_data[:batch_size], x_data[batch_size:]
        y_batch, y_data = y_data[:batch_size], y_data[batch_size:]
        yield x_batch, y_batch
        
        
# Part 3 : Main

data = load_abalone_dataset()
shuffle_data(data)
x_train, y_train, x_test, y_test = train_test_split(data)

network = Network()
trainer = Trainer(network, epoch=10)
trainer.train_generator(minibatch, [x_train, y_train], print_freq=1)
