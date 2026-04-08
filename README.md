---
title: Autoprep Env
emoji: 🐨
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
short_description: RL environment for data cleaning tasks with OpenEnv
---

# AutoPrep-Env: RL Environment for Data Cleaning

## Overview

AutoPrep-Env is a reinforcement learning environment that simulates real-world data cleaning tasks. The environment allows an agent to iteratively improve dataset quality by taking actions such as removing duplicates, filling missing values, and handling outliers.

This environment follows the OpenEnv specification and is designed for evaluating agent decision-making in structured data preprocessing tasks.

---

## Motivation

Data cleaning is a critical step in real-world data pipelines. This environment models the sequential and decision-based nature of cleaning operations, where each action affects the overall data quality.

---

## Environment Design

### Observation Space

The agent observes:

- missing_values: number of missing entries
- duplicate_rows: number of duplicate rows
- outliers: number of outlier values
- step_count: number of steps taken

---

### Action Space

The agent can take the following actions:

- `remove_duplicates`
- `fill_missing`
- `remove_outliers`
- `stop`

---

### Reward Function

The reward is based on:

- Improvement in data quality
- Penalty for unnecessary steps
- Penalty for ineffective actions
- Bonus reward for fully cleaning the dataset

---

## Tasks

Three difficulty levels:

- Easy: small number of issues  
- Medium: moderate dataset issues  
- Hard: large-scale data cleaning challenge  

---

## Grading

Final score is computed between 0.0 and 1.0 based on the remaining issues in the dataset.

---

## How to Run

### Local

```bash
python inference.py