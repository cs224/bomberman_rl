
# agent_010_shred

The basic idea behind this agent is that the complexity of a 15x15 bomberman game field should be comparable to the complexity of 28x28 [MNIST](https://en.wikipedia.org/wiki/MNIST_database) data-set. A good CNN that solves this problem is given here: [mnist-competition](https://github.com/kkweon/mnist-competition), which was the winning model in the [how-far-can-we-go-with-MNIST](https://github.com/hwalsuklee/how-far-can-we-go-with-MNIST) competition. I oriented myself along the simplest of them all, the vgg5 net, because there is anyway a lot of variance in predicting the quality of a move, e.g. you cannot expect too much accuracy in predicting the outcome of moves.

The input data to the agent_010_shred model consists of:
* one-hot encoding of the chosen direction ['A_WAIT', 'A_UP', 'A_LEFT', 'A_DOWN', 'A_RIGHT', 'A_BOMB']
* object_of_interest_data `dx` as (-1,0,1), `dy` as (-1,0,1), `is_direct_path` as (-1,0,1), `path_cost`
  * nearest_other_agent_info_data
  * nearest_coin_info_data
  * nearest_crate_info_data
  * mid_of_map_info_data
  * if there is no (visible) such object on the map then the `is_direct_path` is -1, `path_cost` is -1 and the `dx`, `dy` is 0
  * this data is calculated during the game via a variant of the [a-star](https://www.redblobgames.com/pathfinding/a-star/introduction.html) algorithm.
* central arena view as 11x11 fields translated so that the agent is always in the middle of it.
  * the central arena view is split into 7 layers/channels: `walls`, `crates`, `coins`, `agents`, `bombs`, `btimes`, `explosions`
    * `walls` is 0 for no wall and 1 for walls
    * `crates` is 0 for no wall and 1 for walls
    * `coins` is 0 for no wall and 1 for walls
    * `agents` is 0 for no wall and 1 for walls
    * `bombs` is 0 for no wall and 1 for walls
    * `btimes` is showing the time until a bomb blast will hit the field as a count down from 5 normalized to the interval [0.0,1.0], e.g. possible values are 0.0,0.2,0.4,0.6,0.8,1.0, where 1.0 stands for "currently no bomb in sight"
    * `explosions` is similar like `btimes`, but shows which fields are affected by explosions

The shapes of the two input tensors is `x1`:(, 22), `x2`: (, 7, 11, 11). The input `x2` is then fed into a small [vgg](https://mxnet.incubator.apache.org/_modules/mxnet/gluon/model_zoo/vision/vgg.html) convolutional neutral network (CNN) and the output of this CNN is then flattened and concatenated with `x1` and fed into a 3 layer feed-forward neural network (FNN), which generates the predictions. The FNN uses [dropout](https://medium.com/@amarbudhiraja/https-medium-com-amarbudhiraja-learning-less-to-learn-better-dropout-in-deep-machine-learning-74334da4bfc5) to avoid overfitting.

## Q-Values

Each move generates a Q value for a reward. E.g. destroying a crate is 1 point, collecting a coin is 5 points and killing an opponent is 10 points.
A step also has a `w_step_survived` value (e.g. 0.3), so that survival alone is already valuable. If the agent kills itself it gets a penalty of `-1.0 * (bomb_tmax + 2) * w_step_survived` and if another agent kills the agent then you get a penalty of `-3.0 * w_step_survived`. Waiting moves do not receive a `w_step_survived` value, to discourage a pure waiting strategy.

## QQ-Values (regression target)

After every game a post-processing is happening, where from the end of the game to the start of the game the Q values are summed up but with a discount factor of 0.9. This discounted sum value is called QQ value and it is the regression target. Like that the agent should get an idea of the closer near term future.

In "steady state", e.g. if the agent only continuously survives, the QQ value will be `1/(1-discount_rate) = 10` and the `w_step_survived` value is calibrated so that QQ is 3 in this continuous steady state scenario. Therefore the agent receives a `w_step_survived` of 0.3.

## Learning

The agent then runs in a loop between playing the game and training the neural network. The game run is aiming at generating 100'000 data-sets. This is aided by data multiplication (see below). Roughly after 100 iterations you get some meaningful behaviour. Currently on my equipment a single iteration that generates 100'000 data-sets and trains on them takes 6-8 minutes. So roughly after 10 hours you start to experience some meaningful behaviour, but you're still far away from avoiding really stupid behaviour, e.g. the agent still regularly kills itself in obvious scenarios.

### Data multiplication: rotations and mirrorings

In order to help the training process the data is artificially multiplied by rotating and mirroring the input data, e.g. you can multiply the real game data by a factor of 8 by this.

Originally I also experimented around with even more data generation techniques, e.g. adding moves that would end-up in a sure death scenario or adding data-sets where the agent creates a bomb where it stands, or adding wait moves. But in the end this did not pay off and I dropped that idea.

## Predictions: playing the game

Predictions are generated by taking the game data and multiplying it by the number of possible moves. Then setting the one-hot encoding of a move in each row to one of the possible moves and predicting the potential outcome QQ-value.

Then with a probability of `90%` the action with the maximum predicted QQ-value is taken.
Within the remaining `10%` with a probability of `90%` I sample from the QQ-values with their waiting, e.g. higher QQ-values are more likely.
Within the final remaining `1%` I sample purely randomly.

The most stupid moves (e.g. wasting a move, because you run against a wall or create) are filtered out and not even taken into consideration.

### Multi-armed bandits

I came across [Multi-armed bandits](https://en.wikipedia.org/wiki/Multi-armed_bandit) in other contexts before, like a better way than A/B testing: [When to Run Bandit Tests Instead of A/B/n Tests](https://conversionxl.com/blog/bandit-tests/).

Typically in Q-learning scenarios the [epsilon-greedy strategy](https://en.wikipedia.org/wiki/Multi-armed_bandit#Approximate_solutions) is used, where you simply use raw randomness in a certain fraction of cases to allow the agent to explore other behaviour.

But often a "Bayesian bandit", a.k.a. [Thompson sampling](https://en.wikipedia.org/wiki/Thompson_sampling), strategy works better. There is a recent paper [Efficient Exploration through Bayesian Deep Q-Networks](https://openreview.net/forum?id=B1e7hs05Km) that seems to confirm this. As I did not implement Q-learning strategy by the book I also could not implement the bayesian sampling strategy described in the paper, but I oriented myself along those lines with the above described sampling strategy.

## Batch learning on previously generated Data

My original idea behind not following the [Q-learning algorithm by the book](https://medium.com/@m.alzantot), but rather take the one-hot encoded actions as regression input was the hope that I would be able to use the generated data to later train other agent architectures on that data to speed-up training.

Up to now this did not work a single time. When I take a new agent architecture and train it on the data generated data of a previous agent architecture the metrics like MSE or explained variance are not too bad, but the agent behaviour is awful, e.g. it most often kills itself immediately. My current guess of why this is the case is that each agent makes its own types of mistakes and needs to get punishment and rewards based on its own behaviour.

### Batch learning with reencoding the QQ Values

A special case of the above batch re-learning idea was that I would start out with a discound rate of `0.9`, run 100 iterations or so, then re-encode the QQ values with a higher discount rate, e.g. `0.93`, then `0.95` and finally `0.98`. My hope was that the agent would learn like this to look further into the future and value future rewards more.

But also this special case of batch re-learning did not really work out. The higher the discount rate the higher the variance from a given state to all of its potential future moves becomes, e.g. for the same state the agent either dies in 2 moves or continues to collect rewards for another 30 moves and the QQ values differ largely. At some point the agent just predicts uniform average values for all moves and there is no distinction between "good" and "bad" moves any longer.

## Q-learning like approach

I would call my approach a "Q-learning like approach", because it applies the idea of discounted future rewards.

The longer I watch the behaviour of the agent and its learning success  or mis-successes I realize that there is a very fragile inter relationship between chosing:
* Q Values
* Discount rate
*

# Frameworks

* [tensorflow](https://www.tensorflow.org/)
  * [keras](https://keras.io/)
* [pytorch](https://pytorch.org/)
* [mxnet](https://mxnet.apache.org/)
  * [keras-mxnet](https://github.com/awslabs/keras-apache-mxnet)
  * [gluon](https://gluon.mxnet.io/)
  * [Dive into Deep Learning](https://d2l.ai/)
    * [PDF](https://en.d2l.ai/d2l-en.pdf)


This is my first project working with neural networks. As the whole world is talking about [tensorflow](https://www.tensorflow.org/) I started initially with using tensorflow. But soon I noticed that it is a memory hog, slow and difficult to debug.

I then looked into other directions and made [my own experiments](https://github.com/cs224/mxnet-gluon-vs-sym-and-others), but also found the [Deep Learning Framework Examples](https://github.com/ilkarman/DeepLearningFrameworks) with extensive benchmarks and examples very helpful.

I finally ended up picking mxnet, because it combines the ease of debugging by using the "un-hybridized" gluon interface, with the top speed of "hybridizing" the gluon models. In addition I very much like its "fine-grained composable abstractions" approach, where you can tweak every aspect of it, while still not needing to reinvent the wheel at every turn. If you are used to using keras then keras-mxnet is a drop-in replacement. I finally decided to go with the raw gluon interface to mxnet. Also from a memory consuption perspective mxnet is top notch and uses only small fractions of your GPU memory. I have to say, after my experiences in this project, I have become a fan of mxnet and do not need to look and further.

Also the documentation is top notch, as there is an open book describing gluon in detail: [Dive into Deep Learning](https://d2l.ai/). There is also a [PDF](https://en.d2l.ai/d2l-en.pdf) version of it.
