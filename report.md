# Analysis: How My Neural Network Learned to Prune Itself

## 1. Project Overview
When we build standard AI models, we often give them way more "brain power" (weights) than they actually need. This creates a lot of wasted space and slow computation. 

In this project, I explored a **self-pruning** approach. Instead of me going in and manually cutting out connections after the model was finished training, I designed the network to figure out which parts were useless on its own. It essentially "trimmed its own fat" while it was learning how to recognize images.

## 2. The Strategy: The Gating Mechanism
I implemented a clever trick called a **Gating Mechanism**. Imagine every connection in the brain having a little "on/off" switch controlled by a score.

* **The Gate:** I used a Sigmoid function on each score. This creates a multiplier between 0 (completely off) and 1 (completely on).
* **The Math:** New Weight = Old Weight * Gate. If the network learns that a gate should be 0, that weight is effectively deleted from the system.
* **The Incentive:** To make sure the network didn't just leave every gate "on," I added a **Sparsity Loss**. Think of this as a penalty for being wasteful. It forces the model to be very picky about which connections it keeps active.

## 3. Training and Data
I put this theory to the test using the **CIFAR-10** dataset (a famous collection of 60,000 tiny images like cars, birds, and planes). 

To make the training even harder and more realistic, I used **Data Augmentation**:
* **Horizontal Flips:** Flipping the images sideways.
* **Random Cropping:** Zooming in on different parts of the photo.
This ensures the model isn't just memorizing pictures but is actually learning the important features that allow it to prune correctly.

## 4. Key Performance Metrics
I ran three different tests, increasing the "penalty" (Lambda) for keeping weights each time. 

| Lambda | Test Accuracy (%) | Sparsity (%) |
|--------|------------------|--------------|
| 0.05   | 53.06            | 2.37         |
| 0.10   | 53.86            | 2.48         |
| 0.20   | 54.82            | 4.13         |

## 5. What the Data Tells Us
The most interesting discovery here was the **Sparsity-Accuracy trade-off**. 

* **Better with Less:** Even when we deleted 4.13% of the weights, the accuracy actually went up! This suggests the original model had "dead weight" that was distracting it or causing it to overthink (overfitting). 
* **Clear Decisions:** When I looked at the gate distribution, I found that the model wasn't being indecisive. Most gates were either pushed all the way to 0 or all the way to 1. The model made very clear "binary" choices about which features were important and which were trash.

## 6. Closing Thoughts
This experiment proves that we don't have to manually tune every part of an AI. By giving the model a small penalty for being bulky, it naturally optimizes its own architecture to be leaner and more efficient. 

In the future, I want to keep increasing the penalty to find the "breaking point"—the exact moment where pruning too much finally starts to make the model lose its memory or accuracy.
