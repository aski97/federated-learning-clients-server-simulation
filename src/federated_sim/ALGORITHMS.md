# Federated Aggregation Algorithms

This repository provides a Python implementation of several server-side aggregation algorithms for Federated Learning (FL). These algorithms define the strategy used by the central server to aggregate updates from multiple clients and produce a new global model.

All algorithms are implemented as classes inheriting from the abstract base class `AggregationAlgorithm` and are designed to be used in a modular fashion.

## Core Concepts: Aggregation

Most algorithms operate on a set of client models (or "weights") provided after local training. We define two key terms:

1.  **Global Model ($x_t$):** The state of the global model on the server at the beginning of aggregation round $t$.
2.  **Client Models ($x_{i,K}^t$):** The model weights from client $i$ after it has performed $K$ local training steps (starting from $x_t$).
3.  **Weighted Average ($x_{\text{avg}}^t$):** The weighted average of all client models, typically weighted by the number of training samples ($n_i$) on each client.
    $$
    x_{\text{avg}}^t = \sum_{i \in S} \frac{n_i}{N} x_{i,K}^t \quad \text{where } N = \sum_{i \in S} n_i
    $$
4.  **Pseudo-Gradient ($\Delta_t$):** The core concept used by all adaptive optimizers. It represents the aggregated update (or the average direction) computed by the clients.
    $$
    \Delta_t = x_{\text{avg}}^t - x_t
    $$

## Implemented Algorithms

### 1. FedAvg (Federated Averaging)

* **Class:** `FedAvg`
* **Description:** This is the standard, canonical Federated Learning algorithm. The new global model ($x_{t+1}$) is simply set to be the weighted average of all client models.
* **Formula:**
    $$
    x_{t+1} = x_{\text{avg}}^t = \sum_{i \in S} \frac{n_i}{N} x_{i,K}^t
    $$
* **Source:** McMahan, H. B., Moore, E., Ramage, D., Hampson, S., & y Arcas, B. A. (2017). "Communication-Efficient Learning of Deep Networks from Decentralized Data." *AISTATS*.

### 2. FedSGD (Federated SGD)

* **Class:** `FedSGD`
* **Description:** An alternative to FedAvg where clients compute and return gradients ($g_i^t$) instead of model weights. The server averages these gradients and applies a single standard SGD step to the global model. This algorithm requires clients to return gradients, not weights.
* **Formula:**
    1.  Compute average gradient: $g_t = \sum_{i \in S} \frac{n_i}{N} g_i^t$
    2.  Update global model: $x_{t+1} = x_t - \eta g_t$
        (where $\eta$ is the server-side learning rate)

### 3. FedMiddleAvg

* **Class:** `FedMiddleAvg`
* **Description:** A custom aggregation strategy. It first computes the standard `FedAvg` result ($x_{\text{avg}}^t$) and then takes the arithmetic mean of that result and the previous global model ($x_t$). This acts as a "damped" update, moving the global model only halfway towards the new FedAvg position.
* **Formula:**
    $$
    x_{t+1} = \frac{x_{\text{avg}}^t + x_t}{2}
    $$

### 4. FedAvgMomentum (Server-Side Momentum)

* **Class:** `FedAvgMomentum`
* **Description:** This algorithm applies momentum on the server to the aggregated updates ($\Delta_t$). It maintains a velocity vector ($m_t$) that accumulates an exponentially moving average of past updates, which helps stabilize training and accelerate convergence.
* **Formula:**
    1.  Compute pseudo-gradient: $\Delta_t = x_{\text{avg}}^t - x_t$
    2.  Update momentum: $m_t = \beta m_{t-1} + (1 - \beta) \Delta_t$
    3.  Update global model: $x_{t+1} = x_t + \eta m_t$
        (where $\eta$ is the server learning rate and $\beta$ is the momentum decay factor)

---

## Adaptive Federated Optimization Algorithms

The following algorithms (`FedAdagrad`, `FedYogi`, `FedAdam`) are based on the paper **"ADAPTIVE FEDERATED OPTIMIZATION"**. They implement server-side adaptive optimizers by treating the aggregated pseudo-gradient $\Delta_t$ as the gradient input for an Adam-like update rule.

* **Source:** Reddi, S., Charles, Z., Zaheer, M., Garrett, Z., Rush, K., Konecny, J., ... & McMahan, H. B. (2020). "Adaptive Federated Optimization." *arXiv:2003.00295*.

All three algorithms share the same update for the 1st moment (momentum) $m_t$ and the same final update step, differing only in how they compute the 2nd moment $v_t$.

**Common Steps:**
1.  **Compute 1st Moment:** $m_t = \beta_1 m_{t-1} + (1 - \beta_1)\Delta_t$
2.  **Bias-Correct 1st Moment:** $\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$
3.  **Final Model Update:** $x_{t+1} = x_t + \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t + \epsilon}}$
    *(Note: For FedAdagrad, $v_t$ is used directly without bias correction $\hat{v}_t$)*

### 5. FedAdagrad

* **Class:** `FedAdagrad`
* **Description:** Adapts the Adagrad optimization rule to the server. It accumulates the sum of squares of past pseudo-gradients. This causes the learning rate to monotonically decrease for parameters with large or frequent updates.
* **2nd Moment ($v_t$) Formula:**
    $$
    v_t = v_{t-1} + \Delta_t^2
    $$

### 6. FedAdam

* **Class:** `FedAdam`
* **Description:** Adapts the Adam optimization rule to the server. It maintains an *exponentially moving average* of both the pseudo-gradient (1st moment $m_t$) and its squared values (2nd moment $v_t$).
* **2nd Moment ($v_t$) Formula:**
    $$
    v_t = \beta_2 v_{t-1} + (1 - \beta_2)\Delta_t^2
    $$
* **Bias-Correction:** $\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$

### 7. FedYogi

* **Class:** `FedYogi`
* **Description:** A variant of FedAdam proposed in the same paper. It uses a different update rule for $v_t$ that controls its growth more effectively, which can help prevent the variance estimate from becoming too large and improve stability.
* **2nd Moment ($v_t$) Formula:**
    $$
    v_t = v_{t-1} - (1 - \beta_2) \cdot \text{sign}(v_{t-1} - \Delta_t^2) \cdot \Delta_t^2
    $$
* **Bias-Correction:** $\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$

---

## Robust Aggregation Algorithms

These algorithms are designed to be resilient to outlier clients, such as those sending corrupted or malicious (Byzantine) updates.

### 8. FedMedian

* **Class:** `FedMedian`
* **Description:** A robust aggregation algorithm that computes the **element-wise median** of client model weights instead of the mean. The median is a robust statistic, making the aggregation step highly resilient to a fraction of outlier clients.
* **Formula:**
    $$
    x_{t+1}[j] = \text{Median}\left( x_{1,K}^t[j], x_{2,K}^t[j], \ldots, x_{|S|,K}^t[j] \right)
    $$
    (where $j$ indexes an individual parameter in the model)

---

## Personalized Federated Learning (PFL) Algorithms

These algorithms aim to create models that are personalized for each client's local data, rather than a single "one-size-fits-all" global model.

### 9. FedRep

* **Class:** `FedRep`
* **Description:** Implements Federated Representation Learning. The model is split into a shared **"base"** (representation) and a personal **"head"** (classifier). The server only aggregates the base layers from clients using FedAvg. The head layers are trained and kept locally by each client and are never sent to the server.
* **Formula:**
    1.  Split model: $x = (x_{\text{base}}, x_{\text{head}})$
    2.  Compute FedAvg for base: $x_{\text{base, avg}}^t = \sum_{i \in S} \frac{n_i}{N} x_{i, \text{base}}^t$
    3.  Update global model: $x_{t+1} = (x_{\text{base, avg}}^t, x_{\text{head}}^t)$
        (The global model's head $x_{\text{head}}^t$ is retained from the previous round. Clients receive the new $x_{\text{base, avg}}^t$ and continue training their personal $x_{i, \text{head}}$).
* **Source:** Collins, L., Hassani, H., Mokhtari, A., & Shakkottai, S. (2021). "Exploiting Shared Representations for Personalized Federated Learning." *ICML*.

---

## Notes on Client-Side Optimization

While this repository focuses on *server-side* aggregation, the choice of the *client-side* optimizer is equally critical.

1.  **Recommended: Standard SGD**
    * The vast majority of FL algorithms, including the original **FedAvg**, assume clients perform their local training using standard **Stochastic Gradient Descent (SGD)**.
    * This is lightweight, computationally cheap, and its (mostly) stateless nature works well with the "averaging" paradigm.

2.  **Caution: Client-Side Adaptive Optimizers (e.g., Adam)**
    * It is generally **not recommended** to use stateful, adaptive optimizers like **Adam**, **RMSProp**, or **Adagrad** on the clients *when using FedAvg*.
    * **Reason:** These optimizers maintain internal states (e.g., 1st and 2nd moment vectors $m$ and $v$ for Adam). When the server performs `FedAvg`, it averages the *model weights* ($x_i$) but not the *optimizer states*.
    * When a client receives the new global model ($x_{t+1}$), its local optimizer state is now "out of sync" with the new weights, leading to instability and poor convergence.
    * **The Solution:** This problem is the entire motivation for the **Adaptive Federated Optimization** algorithms (`FedAdam`, `FedYogi`). They **move the adaptive logic to the server**, allowing clients to remain simple (using SGD) while the global model benefits from adaptive updates.

3.  **For Non-IID Data: FedProx or SCAFFOLD**
    * If you are dealing with significant data heterogeneity (non-IID data), the main challenge is "client drift," where local models diverge too far from the global consensus.
    * **`FedProx`:** This is a client-side modification. The client still uses SGD, but adds a **proximal term** to its local loss function. This term penalizes the local model for moving too far from the global model it started with, effectively limiting client drift.
    * **`SCAFFOLD`:** This is a more advanced (and stateful) algorithm that modifies both the client and server to correct for client drift using control variates. The client optimizer is still based on SGD.

---

## References

1.  Reddi, S., et al. (2020). **"Adaptive Federated Optimization."**
    * *Link:* [https://arxiv.org/pdf/2003.00295](https://arxiv.org/pdf/2003.00295)
2.  McMahan, H. B., et al. (2017). **"Communication-Efficient Learning of Deep Networks from Decentralized Data."**
    * *Link:* [http://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf](http://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf)
3.  Collins, L., et al. (2021). **"Exploiting Shared Representations for Personalized Federated Learning."**
    * *Link:* [http://proceedings.mlr.press/v139/collins21a/collins21a.pdf](http://proceedings.mlr.press/v139/collins21a/collins21a.pdf)