"""
Take in a "true" network, generate a bunch of training samples from it,
and then try to train a corresponding network with gradient descent.

For example,

    python gd_stable/main/data.py --depth 5 --width 32 \
        --samples 10000 --steps 100

will read in the network in ./data/mlp-5-32.pth and generate 10000
input-output pairs according to the network's current weights.

* We fix input sizes to 64 and output sizes to be 1.
* Samples generated are standard normal distributed.

Then a new network is trained with 100 full gradient descent steps.

Prints a loss curve in ./data/plot-5-32.pdf

"""

from absl import app
from absl import flags
import torch
from torch import optim
from torch.nn import functional as F
import numpy as np

from gd_stable.mlp import MLP
from gd_stable.utils import seed_all, timeit, import_matplotlib

flags.DEFINE_integer('depth', 5, 'depth of generated network')
flags.DEFINE_integer('width', 32, 'width of generated network')
flags.DEFINE_integer('samples', 10000, 'number of samples to generate')
flags.DEFINE_integer('steps', 100, 'number of gradient stepts to take')
flags.DEFINE_integer('batch_size', 1024, 'maximum batch size for processing')
flags.DEFINE_float('learning_rate', 0.01, 'learning rate')


def _main(_):
    seed_all(1)

    input_size = 64
    hiddens = [flags.FLAGS.width] * flags.FLAGS.depth
    output_size = 1
    mlp_true = MLP(input_size, hiddens, output_size)
    savefile = './data/mlp-{}-{}.pth'.format(flags.FLAGS.depth,
                                             flags.FLAGS.width)
    mlp_true.load_state_dict(torch.load(savefile))
    print('loaded MLP {} from {}'.format(
        ' -> '.join(map(str, [input_size] + hiddens + [output_size])),
        savefile))

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('yes using GPU')
    else:
        device = torch.device('cpu')
        print('not using GPU')
    mlp_true = mlp_true.to(device)

    param_counts = [p.numel() for p in mlp_true.parameters()]
    num_parameters = sum(param_counts)
    print('num parameters', num_parameters)

    batch_size = 1024
    ns = flags.FLAGS.samples
    with timeit('generating {} samples'.format(ns)):
        inputs = torch.randn((ns, input_size))
        outputs = torch.zeros((ns, output_size))
        for i in range(0, ns, batch_size):
            low = i
            high = min(ns, i + batch_size)
            batch = inputs[low:high].to(device)
            outputs[low:high] = mlp_true(batch)

    mlp_learned = MLP(input_size, hiddens, output_size)
    mlp_learned = mlp_learned.to(device)
    optimizer = optim.SGD(
        mlp_learned.parameters(), lr=flags.FLAGS.learning_rate)

    # smooth_l1_loss (huber) doesn't fix the gradient issue, still...
    lossfn = F.mse_loss
    evaluate_every = max(flags.FLAGS.steps // 100, 1)
    maxsteps = flags.FLAGS.steps
    maxnorm = 1  # np.sqrt(num_parameters)
    train_losses = []
    test_losses = []
    steps = []
    for step in range(1, 1 + maxsteps):
        optimizer.zero_grad()

        loss = 0
        for i in range(0, ns, batch_size):
            low, high = i, min(ns, i + batch_size)
            with torch.no_grad():
                input_batch = inputs[low:high].to(device)
                expected = outputs[low:high].to(device)
            predicted = mlp_learned(input_batch)
            batch_loss = lossfn(
                predicted, expected, size_average=False).squeeze()
            scaling_factor = (high - low) / ns
            batch_loss.backward(torch.ones(()).to(device) * scaling_factor)
            with torch.no_grad():
                loss += batch_loss.cpu().detach().numpy() * scaling_factor

        with torch.no_grad():
            grad = torch.cat(
                tuple(p.grad.data.view(-1) for p in mlp_learned.parameters()))
            gradnorm = torch.norm(grad)

        # limit norm, unfortunately looks essential...
        if gradnorm > maxnorm:
            for p in mlp_learned.parameters():
                p.grad.data /= gradnorm
        optimizer.step()

        if step == 1 or step == maxsteps or step % evaluate_every == 0:
            with torch.no_grad():
                test_loss = 0
                for i in range(0, ns, batch_size):
                    low, high = i, min(ns, i + batch_size)
                    input_batch = torch.randn((ns, input_size), device=device)
                    expected = mlp_true(input_batch)
                    predicted = mlp_learned(input_batch)
                    batch_loss = lossfn(
                        predicted, expected, size_average=False).squeeze()
                    scaling_factor = (high - low) / ns
                    test_loss += batch_loss.cpu().detach().numpy(
                    ) * scaling_factor
            fmt = '{:' + str(len(str(maxsteps))) + 'd}'
            update = ('step ' + fmt + ' of ' + fmt).format(step, maxsteps)
            update += ' train loss {:8.4g}'.format(loss)
            update += ' grad norm (unclipped) {:8.4g}'.format(gradnorm)
            update += ' test loss {:8.4g}'.format(test_loss)
            print(update)
            steps.append(step)
            train_losses.append(loss)
            test_losses.append(test_loss)

    plt = import_matplotlib()
    plt.clf()
    plt.semilogy(steps, train_losses, label='train')
    plt.semilogy(steps, test_losses, label='test', ls='--')
    plt.legend()
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.title('gd with norm clipping (depth {} width {})'.format(
        flags.FLAGS.depth, flags.FLAGS.width))
    plt.savefig('./data/plot-{}-{}.pdf'.format(flags.FLAGS.depth,
                                               flags.FLAGS.width))


if __name__ == '__main__':
    app.run(_main)
