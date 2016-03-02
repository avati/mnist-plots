import numpy as np
import random
import cPickle as pickle
import argparse
from os.path import join as pjoin
import theano
import theano.tensor as T
from utils import load_dataset, floatX
from mlp import DNN, CategoricalMLP
from opt import optimizers, get_opt_fn

# TODO be able to specify nonlinearity (e.g. ReLU)
# Data that load_dataset uses can be found here http://deeplearning.net/data/mnist/mnist.pkl.gz

NUM_CLASSES = 10

class DNN(object):
    def __init__(self, n_in, n_hid, n_out, nlayers, optimizer):

        self.x = T.matrix('x', dtype=floatX)
        self.y = T.vector('y', dtype='int64')  # one-hot vector
        lr = T.scalar(dtype=floatX)

        self.cat_mlp = CategoricalMLP(self.x, n_in, n_hid, n_out, nlayers=nlayers, y=self.y)
        self.cost = self.cat_mlp.cost / self.x.shape[0]
        self.params = self.cat_mlp.params
        self.grad_params = self.params

        self.updates, self.grad_norm, self.param_norm = get_opt_fn(optimizer)(self.cost, self.grad_params, lr)

        self.train = theano.function(
                inputs=[self.x, self.y, lr],
                outputs=self.cost,
                updates=self.updates
        )
        self.test = theano.function(
            inputs=[self.x, self.y],
            outputs=[self.cost, self.cat_mlp.y_pred],
            updates=None
        )

def balanced_subset(data, key, frac):
    key_x, key_y = data[key]

    labels = np.unique(key_y)
    indices = {}
    for i in range(len(labels)):
        indices[i] = np.where(key_y == labels[i])[0]
        cnt = int(frac * indices[i].shape[0])
        indices[i] = np.random.choice(indices[i], cnt, replace=False)

    perm = np.concatenate(indices.values())
    np.random.shuffle(perm)
    key_x = key_x[perm, :]
    key_y = key_y[perm]
    return key_x, key_y

def measure_perf(x, y, model, args):
    total_cost = 0
    total_correct = 0
    total_total = 0
    num_batches = ((x.shape[0] - 1) / args.batch_size) + 1

    for l in xrange(num_batches):
        x_batch = x[l * args.batch_size:(l + 1) * args.batch_size, :]
        y_batch = y[l * args.batch_size:(l + 1) * args.batch_size]
        cost, pred = model.test(x_batch, y_batch)
        correct = np.sum(pred == y_batch)
        total = np.prod(y_batch.shape)
        total_correct = total_correct + correct
        total_total = total_total + total
        total_cost = total_cost + cost
    total_cost = total_cost / num_batches
    perf = total_correct/float(total_total)
    return perf, cost


def mnist(args):
    data = load_dataset(dset='mnist')
    train_x, train_y = balanced_subset(data, 'train', args.data_frac)
    valid_x, valid_y = balanced_subset(data, 'valid', args.data_frac)
    test_x, test_y = data['test']

    num_train_batches = train_x.shape[0] / args.batch_size
    num_valid_batches = valid_x.shape[0] / args.batch_size
    valid_freq = num_train_batches

    model = DNN(train_x.shape[1], args.hdim, NUM_CLASSES, args.nlayers, args.optimizer)

    expcost = None

    vcosts=[]
    perfs = []
    model_best = None
    prev_perf = 0.0
    perf = 0.0
    for b in xrange(args.epochs * num_train_batches):
        k = b % num_train_batches
        x = train_x[k * args.batch_size:(k + 1) * args.batch_size, :]
        y = train_y[k * args.batch_size:(k + 1) * args.batch_size]
        cost = model.train(x, y, args.lr)
        if not expcost:
            expcost = cost
        else:
            expcost = 0.01 * cost + 0.99 * expcost
        if (b + 1) % args.print_every == 0:
            print('iter %d, cost %f, expcost %f' % (b + 1, cost, expcost))
        if (b + 1) % valid_freq == 0:
            perf, _ = measure_perf(valid_x, valid_y, model, args)
            perfs.append(perf)
            print('correct/total: %f' % (perf))
            if len(perfs) > 32:
                old_perf = perfs.pop(0)
                max_perf = max(perfs)
                if old_perf >= max_perf:
                    print('Peak perf: %f (hdim=%d, nlayers=%d, data_frac=%f, lr=%f)' % (old_perf, args.hdim, args.nlayers, args.data_frac, args.lr))
                    test_perf, _ = measure_perf(test_x, test_y, model, args)
                    print('Test perf: %f (hdim=%d, nlayers=%d, data_frac=%f, lr=%f)' % (test_perf, args.hdim, args.nlayers, args.data_frac, args.lr))
                    return test_perf
                    break
#            perf, vcost = measure_perf(valid_x, valid_y, model, args)
#            vcosts.append(vcost)
#            print('validation perf, cost: %f, %f' % (perf, vcost))
#            if len(vcosts) > 32:
#                old_vcost = vcosts.pop(0)
#                low_vcost = min(vcosts)
#                if old_vcost <= low_vcost:
#                    print('Low vcost: %f (hdim=%d, nlayers=%d, data_frac=%f, lr=%f)' % (old_vcost, args.hdim, args.nlayers, args.data_frac, args.lr))
#                    test_perf, _ = measure_perf(test_x, test_y, model, args)
#                    print('Test perf: %f (hdim=%d, nlayers=%d, data_frac=%f, lr=%f)' % (test_perf, args.hdim, args.nlayers, args.data_frac, args.lr))
#                    return test_perf
#                    break
        if (b + 1) % (num_train_batches * args.save_every) == 0:
            if perf > prev_perf:
                prev_perf = perf
                print('saving model')
                with open(pjoin(args.expdir, 'model.pk'), 'wb') as f:
                    pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

    # XXX just pickling the entire model for now
    if perf > prev_perf:
        print('saving final model')
        with open(pjoin(args.expdir, 'model.pk'), 'wb') as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--nlayers', default=1, type=int, help='number of hidden layers in MLP before output layers')
    parser.add_argument('--hdim', default=500, type=int, help='dimension of hidden layer')
    parser.add_argument('--data_frac', default=1.00, type=float, help='fraction of training data to use')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--epochs', default=100000, type=int, help='number of passes over dataset')
    parser.add_argument('--print_every', default=100, type=int, help='how often to print cost')
    parser.add_argument('--save_every', default=1, type=int, help='how often to save model (in terms of epochs)')
    parser.add_argument('--optimizer', type=str, default='sgd', choices=optimizers)
    parser.add_argument('--expdir', default='/bak/ug_exps/mnist', help='output file to save model to')
    parser.add_argument('--avgcnt', default=50, type=int, help='number of runs to smooth over')
    args = parser.parse_args()
    print(args)

    meanBestPerf = sum(mnist(args) for i in range(args.avgcnt)) / args.avgcnt
    print('Best perf: %f (hdim=%d, nlayers=%d, data_frac=%f, lr=%f)' % (meanBestPerf, args.hdim, args.nlayers, args.data_frac, args.lr))

if __name__ == '__main__':
    main()
