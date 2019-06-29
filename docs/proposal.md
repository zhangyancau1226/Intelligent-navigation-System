---
layout: default
title: Proposal
---
## Summary of the Project

For our project, we plan to create some simple maze and navigate an agent through it. Rewards will be implemented throughout the maze to produce more complex problems and solutions. So we plan to involve a smart item collection and selection AI. For the running process, no user input is needed for the agent. The agent should find a way to the destination automatically. During the process, the agent should also collect some items smartly and use them to remove the trap on the path to ensure it can arrive the destination with the shortest time, most items collected, and the fewest items used.
 
## AI/ML Algorithms

The algorithm that we intend to use for our project is the Deep Q-learning for Reinforcement Learning, but more algorithms may be involved.
 
## Evaluation Plan

Since reinforcement learning depends on rewards, we will introduce different items with different values in the maze where the agent receives some awards based on item values. At the end of the maze, a score could be assigned to this maze solution based on time and items left. 

Quantitative Evaluation:

- Numerical Metrics: Time to process, number of items collected, number of items used, and the total score. The time to train the AI may also be analyzed.
- Baselines: The agent should find the destination and generate the shortest path correctly and quickly without involving item collection and selection AI.

Qualitative Evaluation:

- Simple Example: In a maze that does not contain any item, the agent should simply use the shortest path to reach the destination.
- Super-Impressive Example: In a complex maze that involves traps and items, the agent should reach the highest score. In other words, it smartly gives up some item to save time, uses some item to remove some traps to save time, and bypass some traps to save items to get the perfect result.

## Appointment with the Instructor

Time: 3:30 pm - Thursday, April 25, 2019

Location: DBH 4204
