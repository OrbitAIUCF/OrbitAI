# OrbitAI: AI-Driven Satellite Collision Prevention Model

![My Image](Diagrams/OrbitAI.png)

**OrbitAI** leverages machine learning to predict potential collisions between satellites and space debris, ensuring the safety and efficiency of space operations. By using real-time data, neural networks, and space simulators, this project aims to enhance satellite autonomy and improve space traffic management.

---

## Data Sources

OrbitAI will rely on a variety of data sources to ensure accurate predictions:

- **Satellite Orbital Data**: Data on the position, velocity, and trajectory of satellites.
   - Primary source for Satellite data: https://www.space-track.org/
- **Space Debris Data**: Information from space agencies and tracking stations about debris, including size, velocity, and orbital details.
   - Primary source of Space Debris data: https://celestrak.org/NORAD/elements/
- **Environmental Factors**: Data on solar radiation, atmospheric drag, and gravitational perturbations that affect satellite paths (May be implemented later).


---

## Neural Networks

To achieve accurate predictions, several types of neural networks will be employed:

- **Recurrent Neural Networks (RNNs)**: For handling time-series data and predicting satellite trajectories over time based on past movement.
- **Long Short Term Memory Model (LSTM)**: For handling the vanishing/exploding gradient problem that traditional RNNs face.
- **Graph Neural Networks (GNNs)**: For modeling relationships between multiple satellites in a constellation, facilitating coordination (May be implemented later).


---

## Training and Techniques

**Training** will focus on the following techniques:

- **Supervised Learning**: Using labeled data from past satellite movements and collision scenarios to train models on predicting future events.
- **Reinforcement Learning**: Applying RL techniques for autonomous satellite maneuvers, allowing satellites to make real-time decisions based on collision risk.
- **Simulation-based Training**: Creating simulated satellite orbits to augment the dataset, particularly for rare events like satellite collisions, to improve model robustness.


---

## Simulation Integration

To validate the trained models and test real-time predictions, **OrbitAI** will integrate with space simulators like Orekit or GMAT (General Mission Analysis Tool). These simulators allow the virtual testing of satellite constellations and debris interactions, providing a controlled environment to evaluate the models' effectiveness.

- **Simulations** will also help refine maneuver strategies by allowing virtual trial-and-error of satellite responses to predicted collisions.


---

## Tools & Utilities

- **CI/CD**: OrbitAI Developers have automated many of the version control processes.
     - When a developer pushes new code or makes a pull request, GitHub creates a fresh Ubuntu virtual machine instance, downloads all required dependencies onto the VM, and checks out the rest of the repository all to perform tests for safe integration.
 
---
