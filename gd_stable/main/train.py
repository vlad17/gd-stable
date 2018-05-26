"""
Take in a "true" network, generate a bunch of training samples from it,
and then try to train a corresponding network with gradient descent.

For example,

    python gd_stable/main/data.py --depth 5 --width 32 \
        --samples 10000 --steps 100

will read in the network in ./data/mlp-5-32.pth and generate 10000
input-output pairs according to the network's current weights.
Samples generated are standard normal distributed.

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
from gd_stable.utils import (seed_all, timeit, import_matplotlib,
                             num_parameters, fromflat, toflat)

flags.DEFINE_integer('depth', 5, 'depth of generated network')
flags.DEFINE_integer('width', 32, 'width of generated network')
flags.DEFINE_string(
    'true_network', None, 'the true network that determines the '
    'correct output values in the regression')
flags.DEFINE_integer('samples', 1024, 'number of samples to generate')
flags.DEFINE_integer('eval_batches', 16,
                     'number of batches for generalization eval')
flags.DEFINE_integer('steps', 100, 'number of gradient stepts to take')
flags.DEFINE_integer('batch_size', 1024, 'maximum batch size for processing')
flags.DEFINE_float('learning_rate', 0.01, 'learning rate')
flags.DEFINE_float('grad_norm_clip', 0, 'if >0, clip gradient to this norm')


def _main(_):
    seed_all(1)

    state_dict = torch.load(flags.FLAGS.true_network)
    input_size = state_dict['input_size']
    output_size = state_dict['output_size']
    mlp_true = MLP(input_size, state_dict['hiddens'], output_size)
    mlp_true.load_state_dict(state_dict['network'])
    print('loaded MLP {} from {}'.format(
        ' -> '.join(
            map(str, [input_size] + state_dict['hiddens'] + [output_size])),
        flags.FLAGS.true_network))

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('yes using GPU')
    else:
        device = torch.device('cpu')
        print('not using GPU')
    mlp_true = mlp_true.to(device)
    print('num parameters', num_parameters(mlp_true))

    # make sure that this is the first random operation in torch so the dataset
    # is always generated deterministically
    batch_size = flags.FLAGS.batch_size
    ns = flags.FLAGS.samples
    with timeit('generating {} samples'.format(ns)):
        inputs = torch.randn((ns, input_size))
        outputs = torch.zeros((ns, output_size))
        for i in range(0, ns, batch_size):
            low = i
            high = min(ns, i + batch_size)
            batch = inputs[low:high].to(device)
            outputs[low:high] = mlp_true(batch)

    hiddens = [flags.FLAGS.width] * flags.FLAGS.depth
    mlp_learned = MLP(input_size, hiddens, output_size)
    mlp_learned = mlp_learned.to(device)
    optimizer = optim.SGD(
        mlp_learned.parameters(), lr=flags.FLAGS.learning_rate)
    print('generated learning MLP {}'.format(
        ' -> '.join(
            map(str, [input_size] + hiddens + [output_size]))))


    # smooth_l1_loss (huber) doesn't fix the gradient issue, still...
    lossfn = F.mse_loss
    evaluate_every = max(flags.FLAGS.steps // 100, 1)
    maxsteps = flags.FLAGS.steps
    maxnorm = flags.FLAGS.grad_norm_clip
    train_losses = []
    test_losses = []
    steps = []
    params = []
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

        if step == 1 or step == maxsteps or step % evaluate_every == 0:
            with torch.no_grad():
                test_loss = 0
                for _ in range(flags.FLAGS.eval_batches):
                    input_batch = torch.randn(
                        (batch_size, input_size), device=device)
                    expected = mlp_true(input_batch)
                    predicted = mlp_learned(input_batch)
                    batch_loss = lossfn(
                        predicted, expected, size_average=False).squeeze()
                    scaling_factor = 1 / flags.FLAGS.eval_batches
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
            params.append(toflat(mlp_learned))

        # step after debug info printed so each param vector corresponds
        # to its own grad norm and losses
        # limit norm, unfortunately looks essential...
        if maxnorm > 0 and gradnorm > maxnorm:
            for p in mlp_learned.parameters():
                p.grad.data *= maxnorm / gradnorm
        optimizer.step()


    # TODO:
    # (3) instead of doing viz in this file after training directly,
    # save the flat param vectors, steps, and losses in a torch state_dict.
    # Then a separate main file should let you plot n curves (from multiple
    # such state_dicts) side by side. Next, yet another separate main file
    # should accept two such state dicts containing the training traces,
    # and plot the 2D (or 1D) parameteric plot of the top principle component
    # of the flat param vector iterates' differences, when combined across
    # the two datasets, i.e., for two runs A, B on the same loss, let their
    # iterates be a0, a1, ... and b0, b1, .... Then do PAC on
    # (a1 - a0, a2 - a1, ... an - a(n-1), b1 - b0, b2 - b1, ...)
    # and plot the two loss curves as well as the loss landscape.

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
    flags.mark_flag_as_required('true_network')
    app.run(_main)
