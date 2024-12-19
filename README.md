# RL-LLM-DT Project

Welcome to the **RL-LLM-DT Project**! This repository contains the code for training, improving, and evaluating AI agents for the game of curling. The project is organized into three main components: **train_curling**, **LLMCurling**, and **curling_battle**. Below, you'll find an overview of each component and instructions for getting started.

---

## Table of Contents
- [RL-LLM-DT Project](#rl-llm-dt-project)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Project Structure](#project-structure)
  - [Getting Started](#getting-started)
    - [1. Clone the Repository](#1-clone-the-repository)
    - [2. Install Dependencies](#2-install-dependencies)
    - [3. Explore the Components](#3-explore-the-components)
  - [Components](#components)
    - [train\_curling](#train_curling)
    - [LLMCurling](#llmcurling)
    - [curling\_battle](#curling_battle)
---

## Overview

The **Curling AI Project** aims to develop AI agents for the game of curling using cutting-edge techniques in reinforcement learning (RL) and large language models (LLMs). The project is divided into three main parts:

1. **train_curling**: Implements distributed Proximal Policy Optimization (PPO) to train an AI agent for curling.
2. **LLMCurling**: Uses LLMs to generate and improve decision tree-based strategies for curling.
3. **curling_battle**: Facilitates the generation of logs for battles between RL-trained models and decision tree-based agents.

---

## Project Structure

The repository is structured as follows:
```plaintext
Curling-AI-Project/
├── train_curling/       # Code for distributed PPO training
├── LLMCurling/          # Code for LLM-based decision tree generation and improvement
├── curling_battle/      # Code for generating battle logs between RL models and decision trees
├── README.md            # Project overview (you are here)
└── requirements.txt     # Python dependencies
```

## Getting Started

To get started with the project, follow these steps:

### 1. Clone the Repository
```bash
git clone https://github.com/Linjunjie99/RL-LLM-DT.git
cd RL-LLM-DT
```
### 2. Install Dependencies
Install the required Python packages:
```bash
pip install -r requirements.txt
```
### 3. Explore the Components
- Check out the train_curling folder to explore the distributed PPO training code.

- Stay tuned for the upcoming release of LLMCurling and curling_battle.
  
## Components
### train_curling
The **train_curling** folder contains the implementation of distributed PPO for training an AI agent to play curling. PPO is a state-of-the-art reinforcement learning algorithm that ensures stable and efficient training.

- Key Features:

  - Distributed training for scalability.

  - Proximal Policy Optimization (PPO) algorithm.


- Status:

  - Code is fully implemented and available in the repository.

### LLMCurling
The **LLMCurling** folder focuses on leveraging Large Language Models (LLMs) to generate and improve decision tree-based strategies for curling. This component explores the intersection of AI and strategic decision-making.

- Key Features:

    - LLM-based code generation for decision trees.

    - Optimization of decision trees for curling strategies.

- Status:

  - Code will be released soon. Stay tuned!

### curling_battle
The ***curling_battle** folder is designed to generate logs for battles between RL-trained models and decision tree-based agents. This component is essential for evaluating the performance of different AI strategies.

- Key Features:

    - Battle simulation between RL models and decision trees.

  - Log generation for analysis and comparison.

- Status:

    - Code will be released soon. Stay tuned!

