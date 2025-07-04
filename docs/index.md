# Home

## Welcome to CleanQRL

[<img src="https://img.shields.io/badge/license-MIT-blue">](https://github.com/fhg-iisb-mki/cleanqrl?tab=License-1-ov-file)
[![docs](https://img.shields.io/github/deployments/vwxyzjn/cleanrl/Production?label=docs&logo=vercel)](https://fhg-iisb-mki.github.io/cleanqrl-docs/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

**CleanQRL** is a Reinforcement Learning library specifically tailored to the subbranch of Quantum Reinforcement Learning and is greatly inspired by the amazing work of **[CleanRL](https://github.com/vwxyzjn/cleanrl)**. Just like the classical analogue, we aim to provide high-quality single-file implementations with research-friendly features. The implementations follow the ideas of **[CleanRL](https://github.com/vwxyzjn/cleanrl)**. They are clean and simple, yet scale nicely through additional features such as **[ray tune](https://docs.ray.io/en/latest/tune/index.html)**. The main features of this repository are


* 📜 Single-file implementations of classical and quantum versions of 4+ Reinforcement Learning agents 
* 💾 Tuned and Benchmarked agents (with available configs)
* 🎮 Integration of [gymnasium](https://gymnasium.farama.org/), [mujoco](https://www.gymlibrary.dev/environments/mujoco/index.html) and [jumanji](https://instadeepai.github.io/jumanji/)
* 📘 Examples on how to enhance the standard QRL agents on a variety of games
* 📈 Tensorboard Logging
* 🌱 Local Reproducibility via Seeding
* 🧫 Experiment Management with [Weights and Biases](https://wandb.ai/site)
* 📊 Easy and straight forward hyperparameter tuning with [ray tune](https://docs.ray.io/en/latest/tune/index.html)

What we are missing compared to **[CleanRL](https://github.com/vwxyzjn/cleanrl)**:

* 💸 Cloud Integration with docker and AWS 
* 📹 Videos of Gameplay Capturing


You can read more about **CleanQRL** in [our upcoming paper]().

## Contact and Community

We want to grow as a community, so feel free to post [Github Issues](https://github.com/fhg-iisb-mki/cleanqrl/issues) and PRs! If you are missing any algorithms or have a specific problem to which you want to tailor your QRL algorithms but fail to do so, you can also create a feature request!

## Citing CleanQRL

If you use **CleanQRL** in your work, please cite our [paper]:

Coming Soon!

## Citing CleanRL

If you used mainly the classical parts of our code in your work, please cite the original [CleanRL paper](https://www.jmlr.org/papers/v23/21-1342.html):

```bibtex
@article{huang2022cleanrl,
  author  = {Shengyi Huang and Rousslan Fernand Julien Dossa and Chang Ye and Jeff Braga and Dipam Chakraborty and Kinal Mehta and João G.M. Araújo},
  title   = {CleanRL: High-quality Single-file Implementations of Deep Reinforcement Learning Algorithms},
  journal = {Journal of Machine Learning Research},
  year    = {2022},
  volume  = {23},
  number  = {274},
  pages   = {1--18},
  url     = {http://jmlr.org/papers/v23/21-1342.html}
}
```

