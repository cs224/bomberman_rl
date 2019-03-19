
# agent_011_shred

The basic idea is exactly the same as for [agent_010_shred](https://github.com/cs224/bomberman_rl/tree/master/agent_code/agent_010_shred), so click on the link to its page and read the big picture there: [agent_010_shred](https://github.com/cs224/bomberman_rl/tree/master/agent_code/agent_010_shred).

After implementing [q-learning by the book](https://towardsdatascience.com/cartpole-introduction-to-reinforcement-learning-ed0eb5b58288) for `agent_010_shred` and seeing that no matter what the game performance never reached a really good level I decided to rework the neural network architecture.
The basic architecture is the same as for `agent_010_shred`, except that I made it a bit deeper and did not use dropout layers. FIn addition I used the [ELU](https://mxnet.incubator.apache.org/api/python/gluon/nn.html#mxnet.gluon.nn.ELU) activation function rather than [ReLU](https://mxnet.incubator.apache.org/api/python/ndarray/ndarray.html#mxnet.ndarray.Activation). Furthermore, I noticed that [normalization is essential](https://mlexplained.com/2018/01/10/an-intuitive-explanation-of-why-batch-normalization-really-works-normalization-in-deep-learning-part-1/) and I used [BatchNorm](https://mxnet.incubator.apache.org/api/python/gluon/nn.html#mxnet.gluon.nn.BatchNorm) in every layer. `agent_010_shred` already used batch normalization, but only in the convolutional layers.

You can see a visualization of the architecture here: [agent_011_shred/0000-network-structure-visualization.ipynb](https://nbviewer.jupyter.org/github/cs224/bomberman_rl/blob/master/agent_code/agent_011_shred/0000-network-structure-visualization.ipynb?flush_cache=true).

# Video

[Bomberman Reinforcement Learning: Replay 2019 03 19 07 57 59 video](https://youtu.be/bC2APj4xf_0)

The yellow and green agents are the new `agent_011_shred` agents. The red one is `agent_010_shred` for comparison. There is also one `simple_agent` in the game for reference.

# Problems

While the performance of the agent is by now quite nice I run into problems during training. No matter what I do, e.g. which optimizer I use or which learning rate I configure, the training process runs now always into situations where it ends up delivering `NaN` values. I guess that there are some diverging gradients reached. I tried to also counteract this situation by replacing the `L2` loss by a `HuberLoss`, but this did also not help.

This is really a pity, because the agent up to this point continuously improved its game performance. Not sure what to do about it. Perhaps replace again the `ELU` activation with `ReLU`?
