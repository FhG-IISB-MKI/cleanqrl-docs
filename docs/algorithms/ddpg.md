# DDPG

[Deep Deterministic Policy Gradient (DDPG)](https://arxiv.org/pdf/1509.02971.pdf) is a powerful actor-critic algorithm designed for environments with continuous action spaces. It combines the strengths of deterministic policy gradients and Q-learning, enabling effective learning in high-dimensional control tasks. DDPG maintains separate networks for the actor and critic, along with their respective target networks, which contribute to training stability. It leverages experience replay and soft target updates to reduce variance and improve sample efficiency. Although more sensitive to hyperparameters compared to some newer methods, DDPG remains a strong baseline in continuous control benchmarks. Our implementation closely follows the design and structure outlined by CleanRL.

## Continuous state - continuous action    

The [```ddpg_classical.py```](https://github.com/fhg-iisb-mki/cleanqrl/blob/main/cleanqrl/ddpg_classical.py) and the [```ddpg_quantum.py```](https://github.com/fhg-iisb-mki/cleanqrl/blob/main/cleanqrl/ddpg_quantum.py) have the following features:

* ✅ Work with continuous observation space 
* ✅ Work with continuous action space
* ✅ Work with envs like [Pendulum-v1](https://gymnasium.farama.org/environments/classic_control/pendulum/)
* ✅ Multiple Vectorized Environments 
* ✅ Single file implementation 

### Implementation details

The key difference between the classical and the quantum algorithm's is the ```ddpgAgentQuantum``` class, as shown below

<div style="display: flex;">
  <span style="width: 50%;">
    ```py title="ddpg_quantum.py" linenums="1"
    class ddpgAgentQuantum(nn.Module):
        def __init__(self, observation_size, num_actions, config):
            super().__init__()
            self.config = config
            self.observation_size = observation_size
            self.num_actions = num_actions
            self.num_qubits = config["num_qubits"]
            self.num_layers = config["num_layers"]
            # input and output scaling are always initialized as ones
            self.input_scaling = nn.Parameter(
                torch.ones(self.num_layers, self.num_qubits), requires_grad=True
            )
            self.output_scaling = nn.Parameter(
                torch.ones(self.num_actions), requires_grad=True
            )
            # trainable weights are initialized randomly between -pi and pi
            self.weights = nn.Parameter(
                torch.FloatTensor(self.num_layers, self.num_qubits*2)
                .uniform_(-np.pi, np.pi),
                requires_grad=True,
            )
            device = qml.device(config["device"], wires=range(self.num_qubits))
            self.quantum_circuit = qml.QNode(
                parameterized_quantum_circuit,
                device,
                diff_method=config["diff_method"],
                interface="torch",
            )
        
        def get_action_and_logprob(self, x):
            logits = self.quantum_circuit(
                x,
                self.input_scaling,
                self.weights,
                self.num_qubits,
                self.num_layers,
                self.num_actions,
                self.observation_size,
            )
            logits = torch.stack(logits, dim=1)
            logits = logits * self.output_scaling
            probs = Categorical(logits=logits)
            action = probs.sample()
            return action, probs.log_prob(action)
    ```
  </span>
  <span style="width: 51%;">
    ```py title="ddpg_classical.py" linenums="1"
    class ddpgAgentClassical(nn.Module):
        def __init__(self, observation_size, num_actions):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(observation_size, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, num_actions),
            )
        
        def get_action_and_logprob(self, x):
            logits = self.network(x)
            probs = Categorical(logits=logits)
            action = probs.sample()
            return action, probs.log_prob(action)
    ```
  </span>
</div>

Additionally, we also need to specify a function for the ansatz of the parameterized quantum circuit. 

```py title="ddpg_quantum.py" linenums="1"
def parameterized_quantum_circuit(
    x, input_scaling, weights, num_qubits, num_layers, num_actions, observation_size
):
    for layer in range(num_layers):
        for i in range(observation_size):
            qml.RX(input_scaling[layer, i] * x[:, i], wires=[i])

        for i in range(num_qubits):
            qml.RY(weights[layer, i], wires=[i])

        for i in range(num_qubits):
            qml.RZ(weights[layer, i + num_qubits], wires=[i])

        if num_qubits == 2:
            qml.CZ(wires=[0, 1])
        else:
            for i in range(num_qubits):
                qml.CZ(wires=[i, (i + 1) % num_qubits])

    return [qml.expval(qml.PauliZ(wires=i)) for i in range(num_actions)]
```

In our implementation, the mean of the continuous action is based on the expectation value of the parameterized quantum circuit, while the variance is an additional classical trainable parameter. This parameter is also the same for all continuous actions. For additional information we refer to [Variational Quantum Circuit Design for Quantum Reinforcement Learning on Continuous Environments](https://arxiv.org/pdf/2312.13798).

Our implementation implements some key novelties proposed by Skolik et al [Quantum agents in the Gym](https://quantum-journal.org/papers/q-2022-05-24-720/pdf/).

* ```data reuploading```: In our ansatz, the features of the states are encoded via RX rotation gates. Instead of only encoding the features in the first layer, this process is repeated in each layer. This has been shown to improve training performance by increasing the expressivity of the ansatz.
* ```input scaling```: In our implementation, we define another set of trainable parameters that scale the features that are encoded into the quantum circuits. This has also been shown to improve training performance.
* ```output scaling```: In our implementation, we define a final set of hyperparameters that scales the expectation values that the quantum circuit "outputs". This has also been shown to improve training performance.

We also provide the option to select different ```learning rates``` for the different parameter sets:

```py title="ddpg_quantum.py"
    optimizer = optim.Adam(
        [
            {"params": agent.input_scaling, "lr": lr_input_scaling},
            {"params": agent.output_scaling, "lr": lr_output_scaling},
            {"params": agent.weights, "lr": lr_weights},
        ]
    )
```

Also, you can use a faster pennylane backend for your simulations:

* ```pennylane-lightning```: We enable the use of the ```lightning``` simulation backend by pennylane, which speeds up simulation 

We also add an observation wrapper called ```ArctanNormalizationWrapper``` at the very beginning of the file. Because we encode the features of the states as rotations, we need to ensure that the features are not beyond the interval of - π and π due to the periodicity of the rotation gates. For more details on wrappers, see [Advanced Usage](https://fhg-iisb-mki.github.io/cleanqrl-docs/advanced_usage/jumanji_environments/).


### Experiment results

Coming Soon!



