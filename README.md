# OrbitAI: AI-Driven Satellite Collision Prevention Model

**OrbitAI** leverages machine learning to predict potential collisions between satellites and space debris, ensuring the safety and efficiency of space operations. By using real-time data, neural networks, and space simulators, this project aims to enhance satellite autonomy and improve space traffic management.

---

## Overview

**OrbitAI** focuses on building an AI-powered system for satellite collision prediction. Using machine learning, this project integrates real-time satellite data, environmental perturbations, and space debris tracking to predict the likelihood of collisions. The goal is to enable satellites to autonomously predict and avoid collisions, thereby protecting critical space infrastructure and reducing operational risks.

---

## Data Sources

OrbitAI will rely on a variety of data sources to ensure accurate predictions:

- **Satellite Orbital Data**: Data on the position, velocity, and trajectory of satellites.
   - Primary source for Satellite data: https://www.space-track.org/
- **Space Debris Data**: Information from space agencies and tracking stations about debris, including size, velocity, and orbital details.
   - Primary source of Space Debris data: https://discosweb.esoc.esa.int/
- **Environmental Factors**: Data on solar radiation, atmospheric drag, and gravitational perturbations that affect satellite paths (May be implemented later).

This data will be gathered from open-source repositories, space tracking networks, and agencies such as NASA, ESA, and commercial satellite tracking services.

---

## Neural Networks

To achieve accurate predictions, several types of neural networks will be employed:

- **Recurrent Neural Networks (RNNs)**: For handling time-series data and predicting satellite trajectories over time based on past movement.
- **Long Short Term Memory Model (LSTM) **: For handling the vanishing/exploding gradient problem that traditional RNNs face.
- **Graph Neural Networks (GNNs)**: For modeling relationships between multiple satellites in a constellation, facilitating coordination (May be implemented later).

These networks will be trained to predict satellite positions, velocities, and potential collision zones while considering dynamic factors such as environmental perturbations.

---

## Training and Techniques

**Training** will focus on the following techniques:

- **Supervised Learning**: Using labeled data from past satellite movements and collision scenarios to train models on predicting future events.
- **Reinforcement Learning**: Applying RL techniques for autonomous satellite maneuvers, allowing satellites to make real-time decisions based on collision risk.
- **Simulation-based Training**: Creating simulated satellite orbits to augment the dataset, particularly for rare events like satellite collisions, to improve model robustness.

The training will involve iterative testing using both real and simulated data to optimize the neural networks’ prediction accuracy.

---

## Simulation Integration

To validate the trained models and test real-time predictions, **OrbitAI** will integrate with space simulators like Orekit or GMAT (General Mission Analysis Tool). These simulators allow the virtual testing of satellite constellations and debris interactions, providing a controlled environment to evaluate the models' effectiveness.

Simulations will also help refine maneuver strategies by allowing virtual trial-and-error of satellite responses to predicted collisions.

A custom website for the project may also be developed. The site would have embedded footage of a simulation demonstration. But the main focus is on the development of the model.


