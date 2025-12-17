import numpy as np
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import time
import networkx as nx
from pathlib import Path
from abc import ABC, abstractmethod
from datetime import datetime


class Recommender:
    """
    Recommender system that suggests songs to agents based on novelty and true utility.

    The recommender computes a score for each song as:
    - Without profit: score = novelty_score * true_mean_utility
    - With profit: score = novelty_score * true_mean_utility * profit_coefficient

    where novelty_score = total_listens / song_counts[song]
    (higher score for less-listened songs)

    The recommender always reports the true mean utility to agents, regardless of profit.
    """

    def __init__(self, true_means: np.ndarray, use_profit: bool = False):
        """
        Initialize recommender.

        Parameters:
        -----------
        true_means : np.ndarray
            True mean utilities (m, N) - agent x song
        use_profit : bool
            Whether to include profit coefficients in recommendation scoring
        """
        self.true_means = true_means
        self.m, self.N = true_means.shape
        self.use_profit = use_profit

        # Sample profit coefficients uniformly from [0, 1] if profit is enabled
        if use_profit:
            self.profit_coefficients = np.random.uniform(0, 1, size=self.N)
        else:
            self.profit_coefficients = np.ones(self.N)  # No profit adjustment

    def get_recommendation(self, agent_id: int, song_counts: np.ndarray,
                          total_listens: float) -> tuple:
        """
        Get recommendation for a specific agent.

        Parameters:
        -----------
        agent_id : int
            Agent to recommend for
        song_counts : np.ndarray
            Number of times agent has listened to each song (N,)
        total_listens : float
            Total number of listens by this agent

        Returns:
        --------
        tuple : (song_id, true_mean_utility)
            Recommended song and its true mean utility for this agent
        """
        if total_listens == 0:
            # No history, recommend song with highest true mean
            best_song = np.argmax(self.true_means[agent_id])
            return best_song, self.true_means[agent_id, best_song]

        # Calculate novelty scores for each song
        # novelty = total_listens / song_counts (higher for less-listened songs)
        # Avoid division by zero
        novelty_scores = np.zeros(self.N)
        for song_id in range(self.N):
            if song_counts[song_id] == 0:
                # Never listened to this song - maximum novelty
                novelty_scores[song_id] = total_listens * 10  # High novelty multiplier
            else:
                novelty_scores[song_id] = total_listens / song_counts[song_id]

        # Compute score: novelty * true_mean * profit_coefficient
        scores = novelty_scores * self.true_means[agent_id] * self.profit_coefficients

        # Select song with highest score
        best_song = np.argmax(scores)

        # Return song and its TRUE MEAN (not profit-adjusted score)
        return best_song, self.true_means[agent_id, best_song]


class Agent(ABC):
    """
    Abstract base class for music listener agents.

    Each agent maintains its own listening history and makes decisions about
    which songs to listen to and which other agents to connect with.
    """

    def __init__(self, agent_id: int, num_songs: int):
        """
        Initialize agent.

        Parameters:
        -----------
        agent_id : int
            Unique identifier for this agent
        num_songs : int
            Total number of songs available
        """
        self.agent_id = agent_id
        self.num_songs = num_songs

        # Listening history
        self.song_counts = np.zeros(num_songs, dtype=np.float64)  # Times listened to each song
        self.song_rewards = np.zeros(num_songs, dtype=np.float64)  # Sum of rewards per song
        self.total_listens = 0.0
        self.total_rewards = 0.0

        # Empirical means (cached)
        self.empirical_means = np.zeros(num_songs, dtype=np.float64)
        self.overall_mean = 0.0

    def update_statistics(self, song_id: int, utility: float):
        """Update listening statistics after listening to a song."""
        self.song_counts[song_id] += 1
        self.song_rewards[song_id] += utility
        self.total_listens += 1
        self.total_rewards += utility

        # Update cached means
        self._update_empirical_means()

    def _update_empirical_means(self):
        """Update empirical mean estimates for all songs."""
        with np.errstate(divide='ignore', invalid='ignore'):
            self.empirical_means = np.where(
                self.song_counts > 0,
                self.song_rewards / self.song_counts,
                0.0
            )
            self.overall_mean = self.total_rewards / self.total_listens if self.total_listens > 0 else 0.0

    @abstractmethod
    def select_song(self, neighbors: List['Agent'], correlations: np.ndarray,
                   current_round: int, recommendation: Optional[tuple] = None) -> tuple:
        """
        Select a song to listen to.

        Parameters:
        -----------
        neighbors : List[Agent]
            List of agents this agent is connected to
        correlations : np.ndarray
            Correlation matrix between all agents
        current_round : int
            Current simulation round
        recommendation : Optional[tuple]
            (song_id, reported_utility) from recommender system

        Returns:
        --------
        tuple : (song_id, chose_recommendation)
            Selected song index and whether it was the recommendation
        """
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(id={self.agent_id})"


class UCBCollaborativeAgent(Agent):
    """
    Agent using Upper Confidence Bound with Collaborative Filtering.

    This is the "optimal" agent design from the paper, using UCB exploration
    with collaborative filtering from neighbors.
    """

    def __init__(self, agent_id: int, num_songs: int, exploration_constant: float = 1.0):
        super().__init__(agent_id, num_songs)
        self.c = exploration_constant

    def select_song(self, neighbors: List[Agent], correlations: np.ndarray,
                   current_round: int, recommendation: Optional[tuple] = None) -> tuple:
        """Select song using UCB with collaborative filtering, considering recommendation."""
        ucb_values = np.zeros(self.num_songs)

        for song_k in range(self.num_songs):
            ucb_values[song_k] = self._compute_ucb(
                song_k, neighbors, correlations, current_round
            )

        # If there's a recommendation, add it as a competing option
        if recommendation is not None:
            rec_song_id, rec_utility = recommendation
            # Treat recommendation as having infinite UCB (always competitive)
            # But we want UCB to still potentially win, so we compare fairly
            # The recommendation provides the true mean, so use it directly
            ucb_values[rec_song_id] = max(ucb_values[rec_song_id], rec_utility)

        best_song = np.argmax(ucb_values)

        # Check if we chose the recommendation
        chose_recommendation = False
        if recommendation is not None:
            rec_song_id, _ = recommendation
            chose_recommendation = (best_song == rec_song_id)

        return best_song, chose_recommendation

    def _compute_ucb(self, song_k: int, neighbors: List[Agent],
                    correlations: np.ndarray, current_round: int) -> float:
        """Compute UCB value for a specific song."""
        N_k = self.song_counts[song_k]
        M_k = self._compute_M(song_k, neighbors, correlations)

        # If never explored, return infinity
        if N_k + M_k == 0:
            return float('inf')

        # Empirical mean
        mu_hat_k = self.empirical_means[song_k]

        # Collaborative estimate
        p_hat_k = self._compute_collaborative_estimate(song_k, neighbors, correlations)

        # Weighted mean
        weighted_mean = (N_k * mu_hat_k + M_k * p_hat_k) / (N_k + M_k)

        # Exploration bonus
        if current_round > 0:
            exploration_bonus = self.c * np.sqrt(np.log(current_round) / (N_k + M_k))
        else:
            exploration_bonus = 0.0

        return weighted_mean + exploration_bonus

    def _compute_M(self, song_k: int, neighbors: List[Agent],
                  correlations: np.ndarray) -> float:
        """Compute M_k = sum of |w_ij| for neighbors who have listened to song k."""
        M_k = 0.0
        for neighbor in neighbors:
            if neighbor.song_counts[song_k] > 0:
                M_k += abs(correlations[self.agent_id, neighbor.agent_id])
        return M_k

    def _compute_collaborative_estimate(self, song_k: int, neighbors: List[Agent],
                                       correlations: np.ndarray) -> float:
        """Compute collaborative filtering estimate p̂_k."""
        if len(neighbors) == 0:
            return self.overall_mean

        numerator = 0.0
        denominator = 0.0

        for neighbor in neighbors:
            if neighbor.song_counts[song_k] > 0:
                w_ij = correlations[self.agent_id, neighbor.agent_id]
                mu_hat_jk = neighbor.empirical_means[song_k]
                mu_bar_j = neighbor.overall_mean

                numerator += w_ij * (mu_hat_jk - mu_bar_j)
                denominator += abs(w_ij)

        if denominator > 1e-10:
            return self.overall_mean + numerator / denominator
        else:
            return self.overall_mean


class RandomAgent(Agent):
    """Agent that selects songs uniformly at random."""

    def select_song(self, neighbors: List[Agent], correlations: np.ndarray,
                   current_round: int, recommendation: Optional[tuple] = None) -> tuple:
        """Select a random song (ignores recommendation)."""
        chosen_song = np.random.randint(0, self.num_songs)

        # Check if we randomly chose the recommendation
        chose_recommendation = False
        if recommendation is not None:
            rec_song_id, _ = recommendation
            chose_recommendation = (chosen_song == rec_song_id)

        return chosen_song, chose_recommendation


class GreedyAgent(Agent):
    """Agent that always picks the song with highest empirical mean."""

    def __init__(self, agent_id: int, num_songs: int, epsilon: float = 0.1):
        super().__init__(agent_id, num_songs)
        self.epsilon = epsilon  # Exploration rate

    def select_song(self, neighbors: List[Agent], correlations: np.ndarray,
                   current_round: int, recommendation: Optional[tuple] = None) -> tuple:
        """Select song greedily (with epsilon exploration), considering recommendation."""
        # Epsilon-greedy exploration
        if np.random.random() < self.epsilon:
            chosen_song = np.random.randint(0, self.num_songs)
            chose_recommendation = False
            if recommendation is not None:
                rec_song_id, _ = recommendation
                chose_recommendation = (chosen_song == rec_song_id)
            return chosen_song, chose_recommendation

        # Find songs with listens
        listened = self.song_counts > 0

        # Compare empirical means with recommendation
        if recommendation is not None:
            rec_song_id, rec_utility = recommendation

            if not np.any(listened):
                # No history, take the recommendation
                return rec_song_id, True

            # Compare best empirical mean with recommendation
            best_empirical_song = np.argmax(self.empirical_means)
            best_empirical_mean = self.empirical_means[best_empirical_song]

            # Choose recommendation if its reported utility is higher
            if rec_utility > best_empirical_mean:
                return rec_song_id, True
            else:
                return best_empirical_song, False
        else:
            # No recommendation
            if not np.any(listened):
                return np.random.randint(0, self.num_songs), False

            return np.argmax(self.empirical_means), False


class MusicRecommendationSimulation:
    """
    Vectorized simulation of music listeners forming communities through collaborative filtering.
    Supports heterogeneous agents with different decision algorithms.
    """

    def __init__(
        self,
        agents: List[Agent],
        num_songs: int,
        latent_dim: int = 3,
        sigma_utility: float = 1.0,
        correlation_threshold: float = 0.3,
        edge_removal_patience: int = 3,
        use_recommender: bool = False,
        recommender_profit: bool = False,
        random_seed: int = None
    ):
        """
        Initialize simulation.

        Parameters:
        -----------
        agents : List[Agent]
            List of agent objects (can be different types)
        num_songs : int
            Number of songs
        latent_dim : int
            Number of latent dimensions for preferences
        sigma_utility : float
            Standard deviation for utility sampling
        correlation_threshold : float
            Threshold K for edge creation/removal
        edge_removal_patience : int
            Number of consecutive rounds below threshold before edge removal
        use_recommender : bool
            Whether to use the recommender system
        recommender_profit : bool
            Whether recommender uses profit coefficients (only applies if use_recommender=True)
        random_seed : int, optional
            Random seed for reproducibility
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        self.agents = agents
        self.m = len(agents)  # number of agents
        self.N = num_songs
        self.latent_dim = latent_dim
        self.sigma_utility = sigma_utility
        self.K = correlation_threshold
        self.T_patience = edge_removal_patience
        self.use_recommender = use_recommender
        self.recommender_profit = recommender_profit

        # Initialize songs and agent true preferences from Gaussian mixture
        self.song_qualities = self._sample_gaussian_mixture(num_songs, latent_dim)
        self.agent_preferences = self._sample_gaussian_mixture(self.m, latent_dim)

        # True mean values: μ_{i,j} = v_i · w_j
        self.true_means = self.agent_preferences @ self.song_qualities.T  # (m, N)

        # Initialize recommender if enabled
        self.recommender = None
        if use_recommender:
            self.recommender = Recommender(self.true_means, use_profit=recommender_profit)

        # Tracking variables
        self.current_round = 0

        # Undirected edge tracking (symmetric adjacency matrix)
        self.adjacency_matrix = np.zeros((self.m, self.m), dtype=np.bool_)
        self.edge_below_threshold_count = np.zeros((self.m, self.m), dtype=np.int32)

        # Output directory (will be set during run_simulation)
        self.output_dir = None

        # History tracking
        self.history = {
            'utilities': [],
            'song_choices': [],
            'edge_count': [],
            'correlations': [],
            'regret': [],
            'recommendation_acceptance': [],  # Track which agents chose recommendations
        }

    def _sample_gaussian_mixture(self, n_samples: int, dim: int) -> np.ndarray:
        """Sample from a mixture of 3 Gaussians with same σ but different centers."""
        centers = np.array([
            [-1.0] * dim,
            [0.0] * dim,
            [1.0] * dim
        ])

        sigma = 0.5

        # Vectorized sampling
        center_indices = np.random.randint(0, 3, size=n_samples)
        selected_centers = centers[center_indices]
        noise = np.random.normal(0, sigma, size=(n_samples, dim))
        samples = selected_centers + noise

        return samples

    def get_neighbors(self, agent_id: int) -> List[Agent]:
        """Get list of agents that agent is connected to."""
        neighbor_ids = np.where(self.adjacency_matrix[agent_id])[0]
        return [self.agents[i] for i in neighbor_ids]

    def compute_all_correlations(self) -> np.ndarray:
        """
        Compute Pearson correlation matrix between all agents (vectorized).

        Returns array of shape (m, m) where w[i,j] is correlation between agents i and j.
        """
        # Create mask for songs each agent has listened to
        has_listened = np.zeros((self.m, self.N), dtype=np.bool_)
        empirical_means = np.zeros((self.m, self.N), dtype=np.float64)

        for i, agent in enumerate(self.agents):
            has_listened[i] = agent.song_counts > 0
            empirical_means[i] = agent.empirical_means

        # For each pair of agents, find common songs
        common_songs = has_listened[:, None, :] & has_listened[None, :, :]  # (m, m, N)
        common_count = common_songs.sum(axis=2)  # (m, m)

        # Initialize correlation matrix
        correlations = np.zeros((self.m, self.m))

        # Only compute for pairs with at least one common song
        valid_pairs = common_count > 0

        if not np.any(valid_pairs):
            return correlations

        # Compute correlations
        for i in range(self.m):
            for j in range(i + 1, self.m):
                if valid_pairs[i, j]:
                    common_mask = common_songs[i, j]
                    means_i = empirical_means[i, common_mask]
                    means_j = empirical_means[j, common_mask]

                    if len(means_i) > 0:
                        mean_i = means_i.mean()
                        mean_j = means_j.mean()

                        numerator = ((means_i - mean_i) * (means_j - mean_j)).sum()
                        denom_i = np.sqrt(((means_i - mean_i) ** 2).sum())
                        denom_j = np.sqrt(((means_j - mean_j) ** 2).sum())
                        denominator = denom_i * denom_j

                        if denominator > 1e-10:
                            corr = np.clip(numerator / denominator, -1.0, 1.0)
                            correlations[i, j] = corr
                            correlations[j, i] = corr  # Symmetric

        return correlations

    def sample_utility(self, agent_id: int, song_id: int) -> float:
        """Sample utility for agent listening to song."""
        true_mean = self.true_means[agent_id, song_id]
        utility = np.random.normal(true_mean, self.sigma_utility)
        return utility

    def generate_edge_candidates(self, song_choices: np.ndarray) -> np.ndarray:
        """
        Generate edge candidate matrix.
        If agents i and j listened to the same song, candidate[i,j] = True.
        """
        candidates = song_choices[:, None] == song_choices[None, :]
        np.fill_diagonal(candidates, False)
        return candidates

    def edge_creation_decisions(self, candidates: np.ndarray, correlations: np.ndarray):
        """Process edge creation for all agents."""
        for i in range(self.m):
            agent_candidates = candidates[i]

            if not np.any(agent_candidates):
                continue

            candidate_indices = np.where(agent_candidates)[0]
            candidate_indices = candidate_indices[~self.adjacency_matrix[i, candidate_indices]]

            if len(candidate_indices) == 0:
                continue

            candidate_correlations = correlations[i, candidate_indices]
            above_threshold = candidate_correlations > self.K

            if np.any(above_threshold):
                valid_candidates = candidate_indices[above_threshold]
                valid_correlations = candidate_correlations[above_threshold]
                best_idx = np.argmax(valid_correlations)
                j = valid_candidates[best_idx]

                # Create undirected edge
                self.adjacency_matrix[i, j] = True
                self.adjacency_matrix[j, i] = True

    def edge_removal_decisions(self, correlations: np.ndarray):
        """Process edge removal for all agents."""
        existing_edges = np.triu(self.adjacency_matrix, k=1)
        edge_indices = np.where(existing_edges)

        edges_to_remove = []

        for idx in range(len(edge_indices[0])):
            i, j = edge_indices[0][idx], edge_indices[1][idx]
            w_ij = correlations[i, j]

            if w_ij < self.K:
                self.edge_below_threshold_count[i, j] += 1
                self.edge_below_threshold_count[j, i] += 1

                if self.edge_below_threshold_count[i, j] >= self.T_patience:
                    edges_to_remove.append((i, j))
            else:
                self.edge_below_threshold_count[i, j] = 0
                self.edge_below_threshold_count[j, i] = 0

        for i, j in edges_to_remove:
            self.adjacency_matrix[i, j] = False
            self.adjacency_matrix[j, i] = False
            self.edge_below_threshold_count[i, j] = 0
            self.edge_below_threshold_count[j, i] = 0

    def run_round(self) -> Dict:
        """Execute one round of the simulation."""
        self.current_round += 1

        # 1. Compute correlations between all agents
        correlations = self.compute_all_correlations()

        # 2. Get recommendations for each agent (if recommender enabled)
        recommendations = {}
        if self.recommender is not None:
            for i, agent in enumerate(self.agents):
                rec = self.recommender.get_recommendation(
                    i, agent.song_counts, agent.total_listens
                )
                recommendations[i] = rec

        # 3. Each agent selects a song
        song_choices = np.zeros(self.m, dtype=np.int32)
        chose_recommendation = np.zeros(self.m, dtype=np.bool_)

        for i, agent in enumerate(self.agents):
            neighbors = self.get_neighbors(i)
            rec = recommendations.get(i, None)
            song_id, chose_rec = agent.select_song(neighbors, correlations,
                                                   self.current_round, rec)
            song_choices[i] = song_id
            chose_recommendation[i] = chose_rec

        # 4. Sample utilities and update agent statistics
        utilities = np.zeros(self.m)
        for i, agent in enumerate(self.agents):
            song_k = song_choices[i]
            utility = self.sample_utility(i, song_k)
            utilities[i] = utility
            agent.update_statistics(song_k, utility)

        # 5. Generate edge candidates (only for agents who didn't choose recommendation)
        candidates = self.generate_edge_candidates(song_choices)

        # Filter out agents who chose the recommendation
        for i in range(self.m):
            if chose_recommendation[i]:
                candidates[i, :] = False
                candidates[:, i] = False

        # 6. Edge creation
        edges_before = self.adjacency_matrix.sum() // 2
        self.edge_creation_decisions(candidates, correlations)
        edges_after_creation = self.adjacency_matrix.sum() // 2
        new_edges = edges_after_creation - edges_before

        # 7. Edge removal
        self.edge_removal_decisions(correlations)
        edges_final = self.adjacency_matrix.sum() // 2
        removed_edges = edges_after_creation - edges_final

        # 8. Calculate regret
        optimal_utilities = self.true_means.max(axis=1)
        round_regret = (optimal_utilities - utilities).sum()

        # 9. Store history
        self.history['utilities'].append(utilities.copy())
        self.history['song_choices'].append(song_choices.copy())
        self.history['edge_count'].append(edges_final)
        self.history['correlations'].append(correlations.copy())
        self.history['regret'].append(round_regret)
        self.history['recommendation_acceptance'].append(chose_recommendation.copy())

        # Count recommendation acceptances
        num_accepted = np.sum(chose_recommendation) if self.use_recommender else 0

        return {
            'round': self.current_round,
            'avg_utility': utilities.mean(),
            'edge_count': edges_final,
            'new_edges': new_edges,
            'removed_edges': removed_edges,
            'regret': round_regret,
            'recommendations_accepted': num_accepted,
        }

    def plot_network_graph(self, round_num: int = None, save_path: str = None, show: bool = False):
        """Visualize the community network using NetworkX with weighted edges (Pearson correlation)."""
        # Get current correlations
        if round_num > 20 and round_num % 5 != 0: 
            return
        if len(self.history['correlations']) > 0:
            correlations = self.history['correlations'][-1]
        else:
            correlations = self.compute_all_correlations()

        # Create weighted graph
        G = nx.Graph()

        # Add all nodes
        for i in range(self.m):
            G.add_node(i)

        # Add edges with correlation weights
        edge_weights = {}
        for i in range(self.m):
            for j in range(i + 1, self.m):
                if self.adjacency_matrix[i, j]:
                    weight = correlations[i, j]
                    G.add_edge(i, j, weight=weight)
                    edge_weights[(i, j)] = weight

        plt.figure(figsize=(14, 12))

        pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)

        # Node colors based on agent type
        node_colors = []
        color_map = {
            'UCBCollaborativeAgent': '#1f77b4',  # blue
           # 'RandomAgent': '#ff7f0e',  # orange
            'GreedyAgent': '#2ca02c',  # green
        }

        for agent in self.agents:
            agent_type = agent.__class__.__name__
            node_colors.append(color_map.get(agent_type, '#d62728'))

        # Node sizes based on degree
        degrees = dict(G.degree())
        node_sizes = [400 + 150 * degrees[i] for i in range(self.m)]

        # Draw nodes
        nx.draw_networkx_nodes(G, pos,
                              node_color=node_colors,
                              node_size=node_sizes,
                              alpha=0.9)

        # Draw edges with varying width and color based on correlation
        if len(edge_weights) > 0:
            # Prepare edge properties
            edges = list(edge_weights.keys())
            weights = list(edge_weights.values())

            # Normalize weights for visualization (width)
            min_weight = min(weights) if weights else 0
            max_weight = max(weights) if weights else 1
            weight_range = max_weight - min_weight if max_weight != min_weight else 1

            # Edge widths: scale by correlation strength
            edge_widths = [1 + 4 * abs(w) for w in weights]

            # Edge colors: positive = green, negative = red
            edge_colors = ['green' if w > 0 else 'red' for w in weights]
            edge_alphas = [0.3 + 0.5 * abs(w) for w in weights]

            # Draw edges
            for edge, width, color, alpha in zip(edges, edge_widths, edge_colors, edge_alphas):
                nx.draw_networkx_edges(G, pos,
                                      edgelist=[edge],
                                      width=width,
                                      alpha=alpha,
                                      edge_color=color)

            # Draw edge labels with correlation values
            edge_labels = {edge: f'{weight:.2f}' for edge, weight in edge_weights.items()}
            nx.draw_networkx_edge_labels(G, pos, edge_labels,
                                         font_size=7,
                                         font_color='darkblue',
                                         bbox=dict(boxstyle='round,pad=0.3',
                                                  facecolor='white',
                                                  edgecolor='none',
                                                  alpha=0.7))

        # Node labels showing agent type abbreviation
        labels = {}
        for i, agent in enumerate(self.agents):
            if isinstance(agent, UCBCollaborativeAgent):
                labels[i] = f"U{i}"
            elif isinstance(agent, RandomAgent):
                labels[i] = f"R{i}"
            elif isinstance(agent, GreedyAgent):
                labels[i] = f"G{i}"
            else:
                labels[i] = str(i)

        nx.draw_networkx_labels(G, pos, labels,
                               font_size=9,
                               font_color='white',
                               font_weight='bold')

        # Add legend
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D

        legend_elements = [
            Patch(facecolor=color_map['UCBCollaborativeAgent'], label='UCB Collaborative'),
            #Patch(facecolor=color_map['RandomAgent'], label='Random'),
            Patch(facecolor=color_map['GreedyAgent'], label='Greedy (ε=0.1)'),
            Line2D([0], [0], color='green', linewidth=2, label='Positive Correlation'),
            Line2D([0], [0], color='red', linewidth=2, label='Negative Correlation'),
        ]
        plt.legend(handles=legend_elements, loc='upper right', fontsize=10)

        # Title
        if round_num is not None:
            plt.title(f'Community Network - Round {round_num}\n'
                     f'{self.adjacency_matrix.sum() // 2} edges, '
                     f'{self.m} agents\n'
                     f'Edge weights = Pearson Correlation',
                     fontsize=14, fontweight='bold')
        else:
            plt.title(f'Community Network\n'
                     f'{self.adjacency_matrix.sum() // 2} edges, '
                     f'{self.m} agents\n'
                     f'Edge weights = Pearson Correlation',
                     fontsize=14, fontweight='bold')

        plt.axis('off')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Network graph saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def run_simulation(self, num_rounds: int, verbose: bool = True,
                      plot_every_round: bool = True, save_network_plots: bool = True) -> None:
        """
        Run the simulation for specified number of rounds.

        Parameters:
        -----------
        num_rounds : int
            Number of rounds to simulate
        verbose : bool
            Whether to print progress
        plot_every_round : bool
            Whether to plot network graph at each round
        save_network_plots : bool
            Whether to save network plots to disk
        """
        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"simulation_run_{timestamp}")
        self.output_dir.mkdir(exist_ok=True)

        # Count agent types
        agent_types = {}
        for agent in self.agents:
            agent_type = agent.__class__.__name__
            agent_types[agent_type] = agent_types.get(agent_type, 0) + 1

        print(f"Starting simulation with {self.m} agents and {self.N} songs...")
        print(f"Agent composition: {agent_types}")
        print(f"Parameters: K={self.K}, T={self.T_patience}, σ={self.sigma_utility}")
        print(f"Recommender: {'ENABLED' if self.use_recommender else 'DISABLED'}")
        if self.use_recommender:
            print(f"Recommender Profit Mode: {'ENABLED' if self.recommender_profit else 'DISABLED'}")
        print(f"Output directory: {self.output_dir}/")
        print("=" * 80)

        if save_network_plots and plot_every_round:
            network_plots_dir = self.output_dir / "network_plots"
            network_plots_dir.mkdir(exist_ok=True)
            print(f"Network plots will be saved to '{network_plots_dir}/' at each round")

        total_start_time = time.time()

        for round_num in range(num_rounds):
            stats = self.run_round()

            if verbose:
                rec_info = ""
                if self.use_recommender:
                    rec_info = f" | Recs: {stats['recommendations_accepted']}/{self.m}"

                print(f"Round {stats['round']:3d} | "
                      f"Avg Utility: {stats['avg_utility']:6.3f} | "
                      f"Edges: {stats['edge_count']:3d} (+{stats['new_edges']}, -{stats['removed_edges']}) | "
                      f"Regret: {stats['regret']:7.2f}{rec_info}")

                # Print adjacency matrix
                print("  Adjacency Matrix:")
                print("    ", end="")
                for j in range(self.m):
                    print(f"{j:>3}", end="")
                print()
                for i in range(self.m):
                    print(f"  {i:>2}", end="")
                    for j in range(self.m):
                        print(f"{int(self.adjacency_matrix[i, j]):>3}", end="")
                    print()

            # Plot network at each round
            if plot_every_round:
                if save_network_plots:
                    save_path = self.output_dir / "network_plots" / f"network_round_{round_num + 1:03d}.png"
                    self.plot_network_graph(round_num=round_num + 1,
                                          save_path=str(save_path),
                                          show=False)
                else:
                    self.plot_network_graph(round_num=round_num + 1, show=True)

        total_time = time.time() - total_start_time

        print("=" * 80)
        print(f"Simulation complete after {num_rounds} rounds.")
        print(f"Total time: {total_time:.2f}s")
        print(f"Final edge count: {self.adjacency_matrix.sum() // 2}")
        print(f"Total cumulative regret: {sum(self.history['regret']):.2f}")

        # Plot final network
        print("\nGenerating final network visualization...")
        final_network_path = self.output_dir / "final_network.png"
        self.plot_network_graph(
            round_num=self.current_round,
            save_path=str(final_network_path),
            show=False
        )

        # Export statistics
        print("Exporting agent statistics...")
        stats_path = self.output_dir / "agent_statistics.txt"
        self.export_agent_statistics(str(stats_path))

    def plot_results(self) -> None:
        """Visualize simulation results."""
        # Adjust subplot layout based on whether recommender is enabled
        if self.use_recommender:
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        else:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Average utility over time
        avg_utilities = [u.mean() for u in self.history['utilities']]

        # Calculate optimal average utility (constant line)
        # For each agent, find their best true mean and average across all agents
        optimal_avg_utility = np.mean(self.true_means.max(axis=1))

        axes[0, 0].plot(avg_utilities, linewidth=2, label='Sampled Avg Utility')
        axes[0, 0].axhline(y=optimal_avg_utility, color='red', linestyle='--',
                          linewidth=2, label=f'Optimal Avg Utility ({optimal_avg_utility:.3f})')
        axes[0, 0].set_xlabel('Round')
        axes[0, 0].set_ylabel('Average Utility')
        axes[0, 0].set_title('Average Utility per Round')
        axes[0, 0].legend(loc='best')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Edge count over time
        axes[0, 1].plot(self.history['edge_count'], linewidth=2, color='green')
        axes[0, 1].set_xlabel('Round')
        axes[0, 1].set_ylabel('Number of Edges')
        axes[0, 1].set_title('Community Network Size (Undirected)')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Cumulative regret
        cumulative_regret = np.cumsum(self.history['regret'])
        axes[1, 0].plot(cumulative_regret, linewidth=2, color='red')
        axes[1, 0].set_xlabel('Round')
        axes[1, 0].set_ylabel('Cumulative Regret')
        axes[1, 0].set_title('Cumulative Regret over Time')
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Song diversity
        unique_songs_per_round = [len(np.unique(choices)) for choices in self.history['song_choices']]
        if self.use_recommender:
            ax_diversity = axes[1, 1]
        else:
            ax_diversity = axes[1, 1]

        ax_diversity.plot(unique_songs_per_round, linewidth=2, color='purple')
        ax_diversity.set_xlabel('Round')
        ax_diversity.set_ylabel('Unique Songs Chosen')
        ax_diversity.set_title('Song Diversity per Round')
        ax_diversity.grid(True, alpha=0.3)

        # 5. Recommendation acceptance (only if recommender enabled)
        if self.use_recommender:
            acceptance_per_round = [np.sum(rec) for rec in self.history['recommendation_acceptance']]
            acceptance_rate = [acc / self.m * 100 for acc in acceptance_per_round]

            axes[1, 2].plot(acceptance_rate, linewidth=2, color='orange')
            axes[1, 2].set_xlabel('Round')
            axes[1, 2].set_ylabel('Recommendation Acceptance Rate (%)')
            axes[1, 2].set_title('Recommendation Acceptance Over Time')
            axes[1, 2].set_ylim(0, 100)
            axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save to output directory if it exists
        if self.output_dir is not None:
            results_path = self.output_dir / 'simulation_results.png'
            plt.savefig(str(results_path), dpi=150, bbox_inches='tight')
            print(f"\nPlot saved as '{results_path}'")
        else:
            plt.savefig('simulation_results.png', dpi=150, bbox_inches='tight')
            print("\nPlot saved as 'simulation_results.png'")

        plt.show()

    def get_summary_statistics(self) -> Dict:
        """Get summary statistics of the simulation."""
        return {
            'total_rounds': self.current_round,
            'final_edges': self.adjacency_matrix.sum() // 2,
            'avg_utility': np.mean([u.mean() for u in self.history['utilities']]),
            'total_regret': sum(self.history['regret']),
            'avg_regret_per_round': np.mean(self.history['regret']),
            'final_avg_correlation': np.mean(self.history['correlations'][-1]) if self.history['correlations'] else 0,
        }

    def export_agent_statistics(self, filename: str = "agent_statistics.txt"):
        """
        Export detailed statistics for each agent to a text file.

        Parameters:
        -----------
        filename : str
            Path to output file
        """
        with open(filename, 'w') as f:
            f.write("=" * 100 + "\n")
            f.write("AGENT STATISTICS REPORT\n")
            f.write("=" * 100 + "\n\n")

            f.write(f"Simulation Parameters:\n")
            f.write(f"  Total Rounds: {self.current_round}\n")
            f.write(f"  Number of Agents: {self.m}\n")
            f.write(f"  Number of Songs: {self.N}\n")
            f.write(f"  Correlation Threshold (K): {self.K}\n")
            f.write(f"  Edge Removal Patience (T): {self.T_patience}\n")
            f.write(f"  Utility Std Dev (σ): {self.sigma_utility}\n")
            f.write(f"  Recommender Enabled: {self.use_recommender}\n")
            if self.use_recommender:
                f.write(f"  Recommender Profit Mode: {self.recommender_profit}\n")
            f.write(f"  Final Network Edges: {self.adjacency_matrix.sum() // 2}\n")
            f.write("\n" + "=" * 100 + "\n\n")

            # If recommender with profit is enabled, show profit coefficients
            if self.use_recommender and self.recommender_profit:
                f.write("Song Profit Coefficients:\n")
                f.write("-" * 100 + "\n")
                f.write(f"  {'Song':<6} {'Profit Coeff':<15}\n")
                f.write(f"  {'-'*6} {'-'*15}\n")
                for song_id in range(self.N):
                    profit = self.recommender.profit_coefficients[song_id]
                    f.write(f"  {song_id:<6} {profit:<15.4f}\n")
                f.write("\n" + "=" * 100 + "\n\n")

            # Statistics for each agent
            for i, agent in enumerate(self.agents):
                f.write(f"Agent {i}: {agent.__class__.__name__}\n")
                f.write("-" * 100 + "\n")

                # Agent-specific parameters
                if isinstance(agent, UCBCollaborativeAgent):
                    f.write(f"  Type: UCB Collaborative Agent\n")
                    f.write(f"  Exploration Constant (c): {agent.c}\n")
                elif isinstance(agent, GreedyAgent):
                    f.write(f"  Type: Greedy Agent\n")
                    f.write(f"  Epsilon: {agent.epsilon}\n")
                elif isinstance(agent, RandomAgent):
                    f.write(f"  Type: Random Agent\n")

                f.write(f"  Total Listens: {int(agent.total_listens)}\n")
                f.write(f"  Total Rewards: {agent.total_rewards:.3f}\n")
                f.write(f"  Overall Mean Utility: {agent.overall_mean:.3f}\n")

                # Network information
                neighbors = self.get_neighbors(i)
                neighbor_types = [self.agents[n.agent_id].__class__.__name__ for n in neighbors]
                f.write(f"  Number of Connections: {len(neighbors)}\n")
                if len(neighbors) > 0:
                    f.write(f"  Connected to Agents: {[n.agent_id for n in neighbors]}\n")
                    f.write(f"  Neighbor Types: {neighbor_types}\n")

                f.write("\n")

                # Per-song statistics
                f.write(f"  Per-Song Statistics:\n")
                if self.use_recommender and self.recommender_profit:
                    f.write(f"  {'Song':<6} {'True Mean':<12} {'Empirical Mean':<16} {'Listen Count':<14} {'Total Reward':<15} {'Profit':<10}\n")
                    f.write(f"  {'-'*6} {'-'*12} {'-'*16} {'-'*14} {'-'*15} {'-'*10}\n")
                else:
                    f.write(f"  {'Song':<6} {'True Mean':<12} {'Empirical Mean':<16} {'Listen Count':<14} {'Total Reward':<15}\n")
                    f.write(f"  {'-'*6} {'-'*12} {'-'*16} {'-'*14} {'-'*15}\n")

                for song_id in range(self.N):
                    true_mean = self.true_means[i, song_id]
                    empirical_mean = agent.empirical_means[song_id]
                    listen_count = int(agent.song_counts[song_id])
                    total_reward = agent.song_rewards[song_id]

                    if self.use_recommender and self.recommender_profit:
                        profit = self.recommender.profit_coefficients[song_id]
                        f.write(f"  {song_id:<6} {true_mean:<12.3f} {empirical_mean:<16.3f} "
                               f"{listen_count:<14} {total_reward:<15.3f} {profit:<10.4f}\n")
                    else:
                        f.write(f"  {song_id:<6} {true_mean:<12.3f} {empirical_mean:<16.3f} "
                               f"{listen_count:<14} {total_reward:<15.3f}\n")

                # Summary statistics for this agent
                f.write("\n")
                listened_songs = agent.song_counts > 0
                num_songs_tried = np.sum(listened_songs)
                f.write(f"  Songs Explored: {num_songs_tried}/{self.N}\n")

                if num_songs_tried > 0:
                    # Best song (empirical)
                    best_empirical_song = np.argmax(agent.empirical_means)
                    best_empirical_mean = agent.empirical_means[best_empirical_song]
                    best_empirical_listens = int(agent.song_counts[best_empirical_song])

                    # Best song (true)
                    best_true_song = np.argmax(self.true_means[i])
                    best_true_mean = self.true_means[i, best_true_song]

                    f.write(f"  Best Empirical Song: {best_empirical_song} "
                           f"(mean={best_empirical_mean:.3f}, listens={best_empirical_listens})\n")
                    f.write(f"  Best True Song: {best_true_song} (true_mean={best_true_mean:.3f})\n")

                    # Check if agent found the best song
                    found_best = best_empirical_song == best_true_song
                    f.write(f"  Found Optimal Song: {found_best}\n")

                    # Regret calculation for this agent
                    agent_utilities = [self.history['utilities'][r][i] for r in range(self.current_round)]
                    cumulative_utility = sum(agent_utilities)
                    optimal_cumulative_utility = best_true_mean * self.current_round
                    agent_regret = optimal_cumulative_utility - cumulative_utility
                    f.write(f"  Total Regret: {agent_regret:.3f}\n")
                    f.write(f"  Avg Regret per Round: {agent_regret / self.current_round:.3f}\n")

                # Listening History
                f.write("\n  Listening History (Round-by-Round):\n")
                if self.use_recommender and self.recommender_profit:
                    f.write(f"  {'Round':<7} {'Song':<6} {'Utility':<10} {'True Mean':<12} {'Rec?':<6} {'Profit':<10}\n")
                    f.write(f"  {'-'*7} {'-'*6} {'-'*10} {'-'*12} {'-'*6} {'-'*10}\n")
                elif self.use_recommender:
                    f.write(f"  {'Round':<7} {'Song':<6} {'Utility':<10} {'True Mean':<12} {'Rec?':<6}\n")
                    f.write(f"  {'-'*7} {'-'*6} {'-'*10} {'-'*12} {'-'*6}\n")
                else:
                    f.write(f"  {'Round':<7} {'Song':<6} {'Utility':<10} {'True Mean':<12}\n")
                    f.write(f"  {'-'*7} {'-'*6} {'-'*10} {'-'*12}\n")

                for round_idx in range(self.current_round):
                    song_chosen = self.history['song_choices'][round_idx][i]
                    utility_received = self.history['utilities'][round_idx][i]
                    true_mean = self.true_means[i, song_chosen]

                    # Check if recommendation was accepted
                    chose_rec = ""
                    if self.use_recommender and len(self.history['recommendation_acceptance']) > round_idx:
                        chose_rec = "YES" if self.history['recommendation_acceptance'][round_idx][i] else "NO"

                    if self.use_recommender and self.recommender_profit:
                        profit = self.recommender.profit_coefficients[song_chosen]
                        f.write(f"  {round_idx + 1:<7} {song_chosen:<6} {utility_received:<10.3f} "
                               f"{true_mean:<12.3f} {chose_rec:<6} {profit:<10.4f}\n")
                    elif self.use_recommender:
                        f.write(f"  {round_idx + 1:<7} {song_chosen:<6} {utility_received:<10.3f} "
                               f"{true_mean:<12.3f} {chose_rec:<6}\n")
                    else:
                        f.write(f"  {round_idx + 1:<7} {song_chosen:<6} {utility_received:<10.3f} "
                               f"{true_mean:<12.3f}\n")

                # Summary of recommendation acceptance for this agent
                if self.use_recommender:
                    total_recs_accepted = sum(
                        self.history['recommendation_acceptance'][r][i]
                        for r in range(self.current_round)
                    )
                    acceptance_rate = (total_recs_accepted / self.current_round * 100) if self.current_round > 0 else 0
                    f.write(f"\n  Recommendation Acceptance: {total_recs_accepted}/{self.current_round} "
                           f"({acceptance_rate:.1f}%)\n")

                f.write("\n" + "=" * 100 + "\n\n")

            # Network statistics
            f.write("NETWORK STATISTICS\n")
            f.write("=" * 100 + "\n")

            # Adjacency matrix
            f.write("\nAdjacency Matrix (1 = connected, 0 = not connected):\n")
            f.write("     ")
            for j in range(self.m):
                f.write(f"{j:>4}")
            f.write("\n")

            for i in range(self.m):
                f.write(f"{i:>4} ")
                for j in range(self.m):
                    f.write(f"{int(self.adjacency_matrix[i, j]):>4}")
                f.write("\n")

            # Final correlations
            if len(self.history['correlations']) > 0:
                final_corr = self.history['correlations'][-1]
                f.write("\nFinal Pearson Correlations:\n")
                f.write("     ")
                for j in range(self.m):
                    f.write(f"{j:>7}")
                f.write("\n")

                for i in range(self.m):
                    f.write(f"{i:>4} ")
                    for j in range(self.m):
                        f.write(f"{final_corr[i, j]:>7.3f}")
                    f.write("\n")

            # Recommendation statistics summary (if enabled)
            if self.use_recommender:
                f.write("\nRECOMMENDATION STATISTICS SUMMARY\n")
                f.write("=" * 100 + "\n\n")

                # Overall acceptance rate
                total_recommendations = self.current_round * self.m
                total_accepted = sum(
                    np.sum(self.history['recommendation_acceptance'][r])
                    for r in range(self.current_round)
                )
                overall_acceptance_rate = (total_accepted / total_recommendations * 100) if total_recommendations > 0 else 0

                f.write(f"Overall Recommendation Acceptance:\n")
                f.write(f"  Total Recommendations Made: {total_recommendations}\n")
                f.write(f"  Total Accepted: {total_accepted}\n")
                f.write(f"  Overall Acceptance Rate: {overall_acceptance_rate:.1f}%\n\n")

                # Per-agent acceptance rates
                f.write(f"Per-Agent Acceptance Rates:\n")
                f.write(f"  {'Agent':<8} {'Type':<25} {'Accepted':<10} {'Rate':<8}\n")
                f.write(f"  {'-'*8} {'-'*25} {'-'*10} {'-'*8}\n")

                for i, agent in enumerate(self.agents):
                    agent_accepted = sum(
                        self.history['recommendation_acceptance'][r][i]
                        for r in range(self.current_round)
                    )
                    agent_rate = (agent_accepted / self.current_round * 100) if self.current_round > 0 else 0
                    agent_type = agent.__class__.__name__

                    f.write(f"  {i:<8} {agent_type:<25} {agent_accepted}/{self.current_round:<9} {agent_rate:>6.1f}%\n")

                # Acceptance by round
                f.write(f"\nRecommendation Acceptance by Round:\n")
                f.write(f"  {'Round':<7} {'Accepted':<10} {'Rate':<8}\n")
                f.write(f"  {'-'*7} {'-'*10} {'-'*8}\n")

                for round_idx in range(min(20, self.current_round)):  # Show first 20 rounds
                    round_accepted = np.sum(self.history['recommendation_acceptance'][round_idx])
                    round_rate = (round_accepted / self.m * 100) if self.m > 0 else 0
                    f.write(f"  {round_idx + 1:<7} {round_accepted}/{self.m:<9} {round_rate:>6.1f}%\n")

                if self.current_round > 20:
                    f.write(f"  ... (showing first 20 of {self.current_round} rounds)\n")

                f.write("\n")

            f.write("=" * 100 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 100 + "\n")

        print(f"Agent statistics exported to '{filename}'")


def main():
    """
    Run example simulation with heterogeneous agents.
    """
    num_collab = 5
    num_greedy = 5
    num_songs = 100

    # Create mix of different agent types
    agents = []

    # 5 UCB Collaborative agents
    for i in range(num_collab):
        agents.append(UCBCollaborativeAgent(agent_id=i, num_songs=num_songs, exploration_constant=1.0))

    for i in range(num_greedy):
        agents.append(GreedyAgent(agent_id=i, num_songs=num_songs, epsilon=0.1))

    # for i in range(8, 10):
    #     agents.append(RandomAgent(agent_id=i, num_songs=num_songs))

    # Create simulation
    sim = MusicRecommendationSimulation(
        agents=agents,
        num_songs=num_songs,
        latent_dim=3,
        sigma_utility=1.0,
        correlation_threshold=0.3,
        edge_removal_patience=3,
        use_recommender=True,  # Enable recommender system
        recommender_profit=True,  # Enable profit-based recommendations
        random_seed=42
    )

    # Run simulation (plots at each round and saves to disk)
    sim.run_simulation(
        num_rounds=120,
        verbose=True,
        plot_every_round=True,
        save_network_plots=True
    )

    # Get summary
    summary = sim.get_summary_statistics()
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    for key, value in summary.items():
        print(f"{key:30s}: {value}")

    # Plot results
    sim.plot_results()


if __name__ == "__main__":
    main()
