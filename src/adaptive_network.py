import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


class PhaseReducedNetwork:
    """
    Network of phase oscillators with adaptive coupling based on FHN neurons.
    """

    def __init__(self, N, T, epsilon, gamma, K0=0.1, frequency_spread=0.0):
        """
        Parameters:
        -----------
        N : int
            Number of neurons in the network
        T : float
            Period of oscillation (ms), estimated ~39.44 ms for FHN
        epsilon : float
            Learning rate for synaptic plasticity (time^-1)
        gamma : float
            Decay rate for synaptic weights (time^-1)
        K0 : float
            Initial synaptic weight
        frequency_spread : float
            Standard deviation of frequency heterogeneity (0 = homogeneous)
        """
        self.N = N
        self.T = T

        if frequency_spread > 0.0:
            self.omega = np.random.normal(2 * np.pi / T, frequency_spread, N)
        else:
            self.omega = np.ones(N) * 2 * np.pi / T

        self.epsilon = epsilon
        self.gamma = gamma

        # Initialize phases uniformly around the circle
        self.theta = np.random.uniform(0, 2 * np.pi, N)

        # Initialize synaptic weight matrix (all-to-all connectivity)
        self.K = np.ones((N, N)) * K0
        np.fill_diagonal(self.K, 0)

    def dynamics(self, t, y):
        """
        Combined dynamics for phases and synaptic weights.

        State vector y contains:
        - y[0:N]: phases theta_i
        - y[N:]: flattened synaptic weight matrix K_ij
        """
        theta = y[:self.N]
        K_flat = y[self.N:]
        K = K_flat.reshape((self.N, self.N))

        # Phase dynamics
        dtheta = np.zeros(self.N)
        for i in range(self.N):
            coupling_sum = 0
            for j in range(self.N):
                if i != j:
                    coupling_sum += K[i, j] * np.sin(theta[j] - theta[i])
            dtheta[i] = self.omega[i] + coupling_sum

        # Synaptic dynamics
        dK = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    delta_theta = theta[j] - theta[i]
                    dK[i, j] = self.epsilon * np.cos(delta_theta) - self.gamma * K[i, j]

        dy = np.concatenate([dtheta, dK.flatten()])
        return dy

    def simulate(self, t_max, dt=0.1):
        """
        Simulate the network dynamics.

        Parameters:
        -----------
        t_max : float
            Total simulation time (ms)
        dt : float
            Time step for output (ms)
        """
        y0 = np.concatenate([self.theta, self.K.flatten()])
        t_eval = np.arange(0, t_max, dt)

        print(f"Simulating network of {self.N} neurons for {t_max} ms...")
        sol = solve_ivp(
            self.dynamics,
            (0, t_max),
            y0,
            t_eval=t_eval,
            method="RK45",
            rtol=1e-6,
            atol=1e-8,
        )

        self.t = sol.t
        self.theta_history = sol.y[:self.N, :]
        K_history_flat = sol.y[self.N:, :]
        self.K_history = K_history_flat.reshape((self.N, self.N, len(self.t)))

        print(f"Simulation complete. {len(self.t)} time points saved.")
        return self.t, self.theta_history, self.K_history

    def compute_order_parameter(self):
        """
        Compute Kuramoto order parameter.
        """
        complex_phases = np.exp(1j * self.theta_history)
        order_param = np.abs(np.mean(complex_phases, axis=0))
        return order_param

    def compute_energy(self, K, theta, lambda_param):
        """
        Compute cost functional.
        """
        N = len(theta)
        theta_diff = theta[:, np.newaxis] - theta[np.newaxis, :]

        mask = ~np.eye(N, dtype=bool)
        E_sync = -0.5 * np.sum(K[mask] * np.cos(theta_diff[mask]))
        E_cost = 0.5 * lambda_param * np.sum(K**2)

        return E_sync + E_cost

    def plot_results(self, save_fig=False):
        """
        Generate plots of dynamics.
        """
        fig = plt.figure(figsize=(14, 10))

        # 1. Phase trajectories over time
        ax1 = plt.subplot(3, 2, 1)
        for i in range(min(10, self.N)):
            ax1.plot(self.t, self.theta_history[i, :] % (2 * np.pi), alpha=0.7, linewidth=1)
        ax1.set_xlabel("Time (ms)")
        ax1.set_ylabel("Phase θ")
        ax1.set_title("Phase Evolution")
        ax1.set_ylim([0, 2 * np.pi])
        ax1.set_yticks([0, 2 * np.pi])
        ax1.set_yticklabels(["0", "2π"])
        ax1.grid(True, alpha=0.3)

        # 2. Order parameter
        ax2 = plt.subplot(3, 2, 2)
        R = self.compute_order_parameter()
        ax2.plot(self.t, R, "b-", linewidth=2)
        ax2.set_xlabel("Time (ms)")
        ax2.set_ylabel("Order parameter R")
        ax2.set_title("Network Synchronization")
        ax2.set_ylim([0, 1.1])
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=1.0, color="r", linestyle="--", alpha=0.5, label="Perfect sync")
        ax2.legend()

        # 3. Raster plot
        spike_times = self.detect_spikes(threshold=0.0)
        ax3 = plt.subplot(3, 2, 3)
        for i, spikes in enumerate(spike_times):
            ax3.scatter(spikes, [i] * len(spikes), c="black", s=10, marker=".")
        ax3.set_xlabel("Time (ms)")
        ax3.set_ylabel("Neuron ID")
        ax3.set_title("Raster Plot - Spike Times")
        ax3.set_ylim([-0.5, self.N - 0.5])
        ax3.set_xlim([0, self.t[-1]])
        ax3.grid(True, alpha=0.4, axis="x")

        # 4. Mean coupling strength
        ax4 = plt.subplot(3, 2, 4)
        mean_K = np.mean(self.K_history, axis=(0, 1))
        ax4.plot(self.t, mean_K, "g-", linewidth=2)
        ax4.set_xlabel("Time (ms)")
        ax4.set_ylabel("Mean synaptic weight")
        ax4.set_title("Average Coupling Strength")
        ax4.grid(True, alpha=0.3)

        # 5. Distribution of final phase differences
        ax5 = plt.subplot(3, 2, 5)
        theta_final = self.theta_history[:, -1]
        phase_diffs = []
        for i in range(self.N):
            for j in range(i + 1, self.N):
                diff = np.abs(theta_final[j] - theta_final[i])
                diff = np.minimum(diff, 2 * np.pi - diff)
                phase_diffs.append(diff)
        ax5.hist(phase_diffs, bins=30, color="purple", alpha=0.7, edgecolor="black")
        ax5.set_xlabel("Phase difference")
        ax5.set_ylabel("Number of neuron pairs")
        ax5.set_title("Distribution of Final Phase Differences")
        ax5.axvline(x=0, color="r", linestyle="--", label="Synchrony")
        ax5.axvline(x=np.pi, color="b", linestyle="--", label="Anti-phase")
        ax5.set_xticks([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])
        ax5.set_xticklabels(["0", "π/4", "π/2", "3π/4", "π"])
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # 6. Phase coherence on unit circle
        ax6 = plt.subplot(3, 2, 6, projection="polar")
        times_to_plot = [0, len(self.t) // 2, -1]
        colors = ["red", "orange", "green"]
        labels = ["Initial", "Middle", "Final"]

        for idx, color, label in zip(times_to_plot, colors, labels):
            theta_snapshot = self.theta_history[:, idx]
            ax6.scatter(theta_snapshot, np.ones(self.N), c=color, s=40, alpha=0.6, label=label)

        ax6.set_title("Phase Distribution on Unit Circle")
        ax6.legend(loc="upper left", bbox_to_anchor=(1.1, 1.1))

        plt.tight_layout()

        if save_fig:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            results_dir = os.path.join(base_dir, "results")
            os.makedirs(results_dir, exist_ok=True)
            plt.savefig(
                os.path.join(results_dir, "phase_network_results.png"),
                dpi=300,
                bbox_inches="tight",
            )

        plt.show()

        # Cost functional over time
        lambda_param = self.gamma / self.epsilon
        energy_time = []
        for k in range(len(self.t)):
            K_snapshot = self.K_history[:, :, k]
            theta_snapshot = self.theta_history[:, k]
            E = self.compute_energy(K_snapshot, theta_snapshot, lambda_param)
            energy_time.append(E)
        energy_time = np.array(energy_time)

        plt.figure(figsize=(10, 4))
        plt.plot(self.t, energy_time, "m-", linewidth=2)
        plt.xlabel("Time (ms)")
        plt.ylabel("Cost E(t)")
        plt.title("Network Cost Over Time")
        plt.grid(True, alpha=0.3)

        if save_fig:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            results_dir = os.path.join(base_dir, "results")
            os.makedirs(results_dir, exist_ok=True)
            plt.savefig(
                os.path.join(results_dir, "cost_over_time.png"),
                dpi=300,
                bbox_inches="tight",
            )

        plt.show()

        print("\n=== Simulation Summary ===")
        print(f"Initial order parameter: R(0) = {R[0]:.4f}")
        print(f"Final order parameter: R(T) = {R[-1]:.4f}")
        print(f"Mean final synaptic weight: {mean_K[-1]:.4f}")
        print(f"Max final synaptic weight: {np.max(self.K_history[:, :, -1]):.4f}")
        print(f"Min final synaptic weight: {np.min(self.K_history[:, :, -1]):.4f}")
        print(f"Metabolic cost parameter λ = γ/ε = {lambda_param:.4f}")

    def detect_spikes(self, threshold=0.0):
        """
        Detect spike times by finding when phase crosses threshold.
        """
        spike_times = []
        for i in range(self.N):
            spikes = []
            phases = self.theta_history[i, :]
            for j in range(1, len(self.t)):
                prev_phase = phases[j - 1] % (2 * np.pi)
                curr_phase = phases[j] % (2 * np.pi)

                if ((prev_phase < threshold and curr_phase >= threshold) or
                        (prev_phase > 5.0 and curr_phase < 1.0)):
                    spikes.append(self.t[j])

            spike_times.append(np.array(spikes))

        return spike_times


if __name__ == "__main__":
    N = 100
    T = 39.44
    epsilon = 0.001
    gamma = 0.0003

    print(f"Metabolic cost parameter: λ = {gamma / epsilon:.2f}")

    network = PhaseReducedNetwork(N, T, epsilon, gamma, K0=0.04, frequency_spread=0.0)

    t_max = 10 * T
    network.simulate(t_max, dt=1)
    network.plot_results(save_fig=True)
