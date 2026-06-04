# 第10章 Actor-Critic算法

## 10.1 简介

本书之前的章节讲解了基于值函数的方法（DQN）和基于策略的方法（REINFORCE），其中基于值函数的方法只学习一个价值函数，而基于策略的方法只学习一个策略函数。那么，一个很自然的问题是，有没有什么方法既学习价值函数，又学习策略函数呢？答案就是Actor-Critic。Actor-Critic是囊括一系列算法的整体架构，目前很多高效的前沿算法都属于Actor-Critic算法，本章接下来将会介绍一种最简单的Actor-Critic算法。需要明确的是，Actor-Critic算法本质上是基于策略的算法，因为这一系列算法的目标都是优化一个带参数的策略，只是会额外学习价值函数，从而帮助策略函数更好地学习。

## 10.2 Actor-Critic

回顾一下，在REINFORCE算法中，目标函数的梯度中有一项轨迹回报，用于指导策略的更新。REINFORCE算法用蒙特卡洛方法来估计 \(Q(\theta_{t},a)\) ，能不能考虑拟合一个值函数来指导策略进行学习呢？这正是Actor-Critic算法所做的。在策略梯度中，可以把梯度写成下面这个更加一般的形式：

\[g = \mathbb{E}\left[\sum_{t = 0}^{T}\psi_{t}\nabla_{\theta}\log \pi_{\theta}(a_{t}|a_{t})\right]\]

其中， \(\psi_{t}\) 可以有很多种形式：

1. \(\sum_{t = 0}^{T}\gamma^{t}r_{t}\) ：轨迹的总回报；
2. \(\sum_{t = 0}^{T}\gamma^{t - t}r_{t}\) ：动作 \(a_{t}\) 之后的回报；
3. \(\sum_{t = 0}^{T}\gamma^{t - t}r_{t} - b(a_{t})\) ：基准线版本的改进；
4. \(Q^{\pi_{\theta}}(a_{t},a_{t})\) ：动作价值函数；
5. \(A^{\pi_{\theta}}(a_{t},a_{t})\) ：优势函数；
6. \(r_{t} + \gamma V^{\pi_{\theta}}(a_{t + 1}) - V^{\pi_{\theta}}(a_{t})\) ：时序差分残差。

9.5节提到REINFORCE通过蒙特卡洛采样的方法对策略梯度的估计是无偏的，但是方差非常大。我们可以用形式(3)引入基线函数（baseline function） \(b(a_{t})\) 来减小方差。此外，我们也可以采用Actor-Critic算法估计一个动作价值函数 \(Q\) ，代替蒙特卡洛采样得到的回报，这便是形式(4)。这个时候，我们可以把状态价值函数 \(V\) 作为基线，从 \(Q\) 函数减去这个 \(V\) 函数则得到了 \(A\) 函数，我们称之为优势函数（advantage function），这便是形式(5)。更进一步，我们可以利用 \(Q = r + \gamma V\) 等式得到形式(6)。

本章将着重介绍形式(6)，即通过时序差分残差 \(\psi_{t} = r_{t} + \gamma V^{\pi}(a_{t + 1}) - V^{\pi}(a_{t})\) 来指导策略梯度进行学习。事实上，用 \(Q\) 值或者 \(V\) 值本质上也是用奖励来进行指导，但是用神经网络进行估计的方法可以减小方差、提高鲁棒性。除此之外，REINFORCE算法基于蒙特卡洛采样，只能在序列结束后进行更新，这同时也要求任务具有有限的步数，而Actor-Critic算法则可以在每一步之后都进行更新，并且不对任务的步数做限制。

我们将Actor-Critic分为两个部分：Actor（策略网络）和Critic（价值网络），如图10-1所示。

Actor要做的是与环境交互，并在Critic价值函数的指导下用策略梯度学习一个更好的策略。Critic要做的是通过Actor与环境交互收集的数据学习一个价值函数，这个价值函数会用于判断在当前状态什么动作是好的，什么动作不是好的，进而帮助Actor进行策略更新。

Actor的更新采用策略梯度的原则，那Critic如何更新呢？我们将Critic价值网络表示为 \(\mathbf{V}_{\omega}\) ，参数为 \(\omega\) 。于是，我们可以采取时序差分残差的学习方式，对于单个数据定义如下价值函数的损失函数：

\[\mathcal{L}(\omega) = \frac{1}{2} (\pmb {r} + \gamma \nabla_{\omega}(\pmb{\sigma}_{t + 1}) - \nabla_{\omega}(\pmb{\sigma}_{t}))^{2}\]

与DQN中一样，我们采取类似于目标网络的方法，将上式中 \(\pmb {r} + \gamma \nabla_{\omega}(\pmb{\sigma}_{t + 1})\) 作为时序差分目标，不会产生梯度来更新价值函数。因此，价值函数的梯度为：

\[\nabla_{\omega}\mathcal{L}(\omega) = -(\pmb {r} + \gamma \nabla_{\omega}(\pmb{\sigma}_{t + 1}) - \nabla_{\omega}(\pmb{\sigma}_{t}))\nabla_{\omega}\nabla_{\omega}(\pmb{\sigma}_{t})\]

然后使用梯度下降方法来更新Critic价值网络参数即可。

Actor-Critic算法的具体流程如下：

初始化策略网络参数 \(\theta\) ，价值网络参数 \(\omega\)

for序列 \(\pmb {e} = \mathbf{1}\rightarrow \pmb {E}\mathbf{d}\mathbf{o}\)

用当前策略 \(\pi_{\theta}\) 采样轨迹 \(\{\pmb{\sigma}_{1},\pmb{\sigma}_{1},\pmb{\sigma}_{1},\pmb{\sigma}_{2},\pmb{\sigma}_{2},\dots \}\)

为每一步数据计算： \(\delta_{t} = r_{t} + \gamma \nabla_{\omega}(\pmb{\sigma}_{t + 1}) - \nabla_{\omega}(\pmb{\sigma}_{t})\)

更新价值参数 \(\omega = \omega +\alpha_{\omega}\sum_{t}\delta_{t}\nabla_{\omega}\nabla_{\omega}(\pmb{\sigma}_{t})\)

更新策略参数 \(\theta = \theta +\alpha_{\theta}\sum_{t}\delta_{t}\nabla_{\theta}\log \pi_{\theta}(\pmb{\sigma}_{t}|\pmb{\sigma}_{t})\)

end for