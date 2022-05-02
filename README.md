# Reinforcement learning playground

## Introduction

A series of reinforcement learning examples built on top of [Tensorflow 2](https://www.tensorflow.org/).

Environments are taken from [Gym](https://gym.openai.com/).

Current methods include:
* Cross entropy method
* Value iteration

## Requirements

* Python 3.10
* Conda package manager ([mini-forge](https://github.com/conda-forge/miniforge) recommended)
* (Recommended) A dedicated gpu (supported by tensorflow)

## Installation

* Clone or download as zip
* `conda env create -f environment.yml`

## Usage

* Running an example (e.g. cartpole_cross_entropy)
  * `python -m reinforcementLearningPlayground.cartpole_cross_entropy`
* Running unit tests
  * `python -m unittest discover tests`

## License

This project is licensed under The MIT License (MIT).
