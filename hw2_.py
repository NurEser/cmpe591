import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt
import pickle
import argparse
from homework2 import Hw2Env

# Hyperparameters
N_ACTIONS = 8
GAMMA = 0.99
EPSILON = 1.0
EPSILON_DECAY = 0.9995  # slower decay for better exploration
EPSILON_DECAY_ITER = 5  # decay epsilon more frequently
MIN_EPSILON = 0.05  # lower minimum for more exploration
LEARNING_RATE = 0.0005  # reduce to prevent overestimation
TAU = 0.005  # more frequent soft updates (was 0.001, too infrequent)
BATCH_SIZE = 32
UPDATE_FREQ = 4  # update the network every 4 steps
TARGET_NETWORK_UPDATE_FREQ = 10000  # not used with soft updates
BUFFER_LENGTH = 100000  # moderate buffer: capture diversity without stale data accumulation
N_EPISODES = 1000
MAX_STEPS = 200  # increased from 50 to allow more complex movements

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQNNetwork(nn.Module):
    """Dueling Deep Q Network for state action value prediction using high-level state."""

    def __init__(self, state_dim=6, action_dim=8, hidden_dim=128):  
        super(DQNNetwork, self).__init__()

        # Shared layers
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        # Value stream
        self.value_fc = nn.Linear(hidden_dim, 64)           
        self.value_ln = nn.LayerNorm(64)
        self.value_head = nn.Linear(64, 1)

        # Advantage stream
        self.advantage_fc = nn.Linear(hidden_dim, 64)       
        self.advantage_ln = nn.LayerNorm(64)
        self.advantage_head = nn.Linear(64, action_dim)

    def forward(self, state):
        # Shared layers with layer normalization
        x = torch.relu(self.ln1(self.fc1(state)))
        x = torch.relu(self.ln2(self.fc2(x)))

        # Value stream
        value = torch.relu(self.value_ln(self.value_fc(x)))
        value = self.value_head(value)

        # Advantage stream
        advantage = torch.relu(self.advantage_ln(self.advantage_fc(x)))
        advantage = self.advantage_head(advantage)

        # Combine: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values


class ReplayBuffer:
    """Experience replay buffer for storing and sampling transitions."""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, terminal):
        self.buffer.append((state, action, reward, next_state, terminal))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, terminals = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(terminals))

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """Deep Q Network Agent for object pushing task."""

    def __init__(self, n_actions=8, state_dim=6):
        self.n_actions = n_actions
        self.state_dim = state_dim
        self.epsilon = EPSILON

        # Networks
        self.q_network = DQNNetwork(state_dim, n_actions).to(device)
        self.target_network = DQNNetwork(state_dim, n_actions).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer and loss
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LEARNING_RATE)
        self.loss_fn = nn.SmoothL1Loss()                    

        # Replay buffer
        self.replay_buffer = ReplayBuffer(BUFFER_LENGTH)
        self.update_count = 0

    def select_action(self, state, training=True):
        """Select action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            return np.random.randint(self.n_actions)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.argmax(dim=1).item()

    def store_transition(self, state, action, reward, next_state, terminal):
        """Store transition in replay buffer.
        
        NOTE: 'terminal' should be True only on genuine episode termination,
        NOT on truncation (hitting MAX_STEPS). Truncated episodes still have
        future value and should keep bootstrapping.
        """
        self.replay_buffer.push(state, action, reward, next_state, terminal)

    def train_step(self):
        """Perform a training step using a batch from replay buffer."""
        if len(self.replay_buffer) < BATCH_SIZE:
            return None, None

        # Sample batch
        states, actions, rewards, next_states, terminals = self.replay_buffer.sample(BATCH_SIZE)

        # Clip rewards more aggressively to prevent overestimation
        rewards = np.clip(rewards, -10, 10)  # allow larger range, but still bounded
        
        # Convert to tensors
        states      = torch.FloatTensor(states).to(device)
        actions     = torch.LongTensor(actions).to(device)
        rewards     = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        terminals   = torch.FloatTensor(terminals).to(device)

        # Compute current Q-values
        q_values = self.q_network(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q-values using TRUE Double DQN:
        #   - online network selects the best next action
        #   - target network evaluates that action
        with torch.no_grad():
            # FIX: was using target_network for both selection & evaluation (standard DQN)
            next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)   # online selects
            next_q_values = self.target_network(next_states).gather(1, next_actions).squeeze(1)  # target evaluates
            target_q_values = rewards + (1 - terminals) * GAMMA * next_q_values

        # Compute loss and update
        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)  # More generous gradient clipping
        self.optimizer.step()

        # Compute Q value magnitudes for monitoring
        q_mag = torch.abs(q_values).mean().item()
        target_q_mag = torch.abs(target_q_values).mean().item()

        self.update_count += 1

        # Decay epsilon every EPSILON_DECAY_ITER updates
        if self.update_count % EPSILON_DECAY_ITER == 0:
            self.epsilon = max(MIN_EPSILON, self.epsilon * EPSILON_DECAY)

        # Soft update target network (polyak averaging) - smooth, stable updates
        for param, target_param in zip(self.q_network.parameters(), self.target_network.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

        return loss.item(), (q_mag, target_q_mag)


def train_dqn():
    """Train DQN agent on object pushing task."""
    env = Hw2Env(n_actions=N_ACTIONS, render_mode="offscreen")
    agent = DQNAgent(n_actions=N_ACTIONS, state_dim=6)

    episode_rewards  = []
    episode_lengths  = []
    episode_successes = []
    q_magnitudes     = []
    losses           = []
    step_counter     = 0
    total_success    = 0
    total_truncated  = 0

    for episode in range(N_EPISODES):
        env.reset()
        state = env.high_level_state()
        done  = False
        cumulative_reward = 0.0
        episode_steps     = 0
        episode_success   = False

        while not done and episode_steps < MAX_STEPS:
            action = agent.select_action(state, training=True)
            _, reward, is_terminal, is_truncated = env.step(action)
            next_state = env.high_level_state()
            done = is_terminal or is_truncated

            if is_terminal:
                episode_success = True
                total_success  += 1
            if is_truncated:
                total_truncated += 1

            # FIX: pass only is_terminal (not done) so truncated episodes
            # still bootstrap instead of incorrectly zeroing future value
            agent.store_transition(state, action, reward, next_state, is_terminal)

            step_counter += 1
            if step_counter % UPDATE_FREQ == 0:
                result = agent.train_step()
                if result[0] is not None:
                    loss, q_mags = result
                    losses.append(loss)
                    q_magnitudes.append(q_mags)

            cumulative_reward += reward
            episode_steps     += 1
            state = next_state

        episode_rewards.append(cumulative_reward)
        episode_lengths.append(episode_steps)
        episode_successes.append(episode_success)

        avg_reward  = np.mean(episode_rewards[-10:])
        avg_length  = np.mean(episode_lengths[-10:])
        avg_rps     = np.mean([r / l for r, l in zip(episode_rewards[-10:], episode_lengths[-10:]) if l > 0])
        success_rate = np.mean(episode_successes[-10:])

        if q_magnitudes:
            recent_q_mags    = q_magnitudes[-20:]
            avg_q_mag        = np.mean([q[0] for q in recent_q_mags])
            avg_target_q_mag = np.mean([q[1] for q in recent_q_mags])
            print(f"Episode {episode+1}/{N_EPISODES} | Avg Reward: {avg_reward:.4f} | "
                  f"Avg Steps: {avg_length:.1f} | Avg RPS: {avg_rps:.4f} | "
                  f"Success Rate: {success_rate:.1%} | Epsilon: {agent.epsilon:.4f}")
            print(f"  Q-Value Mag: {avg_q_mag:.4f} | Target Q-Value Mag: {avg_target_q_mag:.4f}")
        else:
            print(f"Episode {episode+1}/{N_EPISODES} | Avg Reward: {avg_reward:.4f} | "
                  f"Avg Steps: {avg_length:.1f} | Avg RPS: {avg_rps:.4f} | "
                  f"Success Rate: {success_rate:.1%} | Epsilon: {agent.epsilon:.4f}")

        if (episode + 1) % 10 == 0:
            torch.save(agent.q_network.state_dict(), f"dqn_checkpoint_ep{episode+1}.pt")
            print(f"Checkpoint saved at episode {episode+1}")

    print("\n" + "="*60)
    print("Training completed!")
    print(f"Total Successful Episodes (terminal): {total_success}")
    print(f"Total Truncated Episodes: {total_truncated}")
    print(f"Success Rate: {total_success}/{N_EPISODES} = {total_success/N_EPISODES:.1%}")
    print("="*60)

    stats = {
        'episode_rewards':   episode_rewards,
        'episode_lengths':   episode_lengths,
        'episode_successes': episode_successes,
        'q_magnitudes':      q_magnitudes,
        'losses':            losses,
        'total_success':     total_success,
        'total_truncated':   total_truncated,
        'hyperparameters': {
            'GAMMA':                      GAMMA,
            'EPSILON':                    EPSILON,
            'EPSILON_DECAY':              EPSILON_DECAY,
            'EPSILON_DECAY_ITER':         EPSILON_DECAY_ITER,
            'MIN_EPSILON':                MIN_EPSILON,
            'LEARNING_RATE':              LEARNING_RATE,
            'BATCH_SIZE':                 BATCH_SIZE,
            'UPDATE_FREQ':                UPDATE_FREQ,
            'TARGET_NETWORK_UPDATE_FREQ': TARGET_NETWORK_UPDATE_FREQ,
            'BUFFER_LENGTH':              BUFFER_LENGTH,
            'N_EPISODES':                 N_EPISODES,
            'N_ACTIONS':                  N_ACTIONS,
        }
    }

    with open('dqn_training_stats.pkl', 'wb') as f:
        pickle.dump(stats, f)
    print("Training stats saved as 'dqn_training_stats.pkl'")

    return agent, episode_rewards, episode_lengths, episode_successes, q_magnitudes, losses


def load_training_stats(filepath='dqn_training_stats.pkl'):
    """Load saved training statistics."""
    with open(filepath, 'rb') as f:
        stats = pickle.load(f)
    return stats


def load_checkpoint(checkpoint_path, stats_path='dqn_training_stats.pkl'):
    """
    Load a trained agent from checkpoint and restore training state.
    
    Args:
        checkpoint_path (str): Path to the saved model weights (e.g., 'dqn_checkpoint_ep500.pt')
        stats_path (str): Path to saved training statistics
        
    Returns:
        agent (DQNAgent): Agent with loaded weights
        stats (dict): Training statistics including episode count and metrics
        start_episode (int): Episode number to resume from
    """
    # Create agent
    agent = DQNAgent(n_actions=N_ACTIONS, state_dim=6)
    
    # Load model weights
    if os.path.exists(checkpoint_path):
        agent.q_network.load_state_dict(torch.load(checkpoint_path, map_location=device))
        agent.target_network.load_state_dict(agent.q_network.state_dict())
        print(f"✓ Model weights loaded from: {checkpoint_path}")
    else:
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        return None, None, None
    
    # Load training statistics
    stats = None
    start_episode = 0
    if os.path.exists(stats_path):
        stats = load_training_stats(stats_path)
        start_episode = len(stats['episode_rewards'])
        print(f"✓ Training stats loaded from: {stats_path}")
        print(f"  Resuming from episode {start_episode + 1}")
    else:
        print(f"✗ Stats file not found: {stats_path} (Starting fresh stats)")
        stats = {
            'episode_rewards':   [],
            'episode_lengths':   [],
            'episode_successes': [],
            'q_magnitudes':      [],
            'losses':            [],
            'total_success':     0,
            'total_truncated':   0,
        }
    
    return agent, stats, start_episode


def continue_training(checkpoint_path, stats_path='dqn_training_stats.pkl', additional_episodes=500, render_mode='offscreen'):
    """
    Continue training from a loaded checkpoint.
    
    Args:
        checkpoint_path (str): Path to the saved model weights
        stats_path (str): Path to saved training statistics
        additional_episodes (int): Number of additional episodes to train
        render_mode (str): 'gui' or 'offscreen'
    """
    # Load checkpoint and stats
    agent, stats, start_episode = load_checkpoint(checkpoint_path, stats_path)
    
    if agent is None:
        print("Failed to load checkpoint. Exiting.")
        return
    
    # Initialize previous lists
    episode_rewards  = stats['episode_rewards']
    episode_lengths  = stats['episode_lengths']
    episode_successes = stats['episode_successes']
    q_magnitudes     = stats['q_magnitudes']
    losses           = stats['losses']
    total_success    = stats['total_success']
    total_truncated  = stats['total_truncated']
    
    # Resume training
    env = Hw2Env(n_actions=N_ACTIONS, render_mode=render_mode)
    step_counter = 0
    total_new_episodes = start_episode + additional_episodes
    
    print(f"\nResuming training from episode {start_episode + 1} to {total_new_episodes}...")
    print("="*60)
    
    for episode in range(start_episode, total_new_episodes):
        env.reset()
        state = env.high_level_state()
        done  = False
        cumulative_reward = 0.0
        episode_steps     = 0
        episode_success   = False

        while not done and episode_steps < MAX_STEPS:
            action = agent.select_action(state, training=True)
            _, reward, is_terminal, is_truncated = env.step(action)
            next_state = env.high_level_state()
            done = is_terminal or is_truncated

            if is_terminal:
                episode_success = True
                total_success  += 1
            if is_truncated:
                total_truncated += 1

            agent.store_transition(state, action, reward, next_state, is_terminal)

            step_counter += 1
            if step_counter % UPDATE_FREQ == 0:
                result = agent.train_step()
                if result[0] is not None:
                    loss, q_mags = result
                    losses.append(loss)
                    q_magnitudes.append(q_mags)

            cumulative_reward += reward
            episode_steps     += 1
            state = next_state

        episode_rewards.append(cumulative_reward)
        episode_lengths.append(episode_steps)
        episode_successes.append(episode_success)

        avg_reward  = np.mean(episode_rewards[-10:])
        avg_length  = np.mean(episode_lengths[-10:])
        avg_rps     = np.mean([r / l for r, l in zip(episode_rewards[-10:], episode_lengths[-10:]) if l > 0])
        success_rate = np.mean(episode_successes[-10:])

        if q_magnitudes:
            recent_q_mags    = q_magnitudes[-20:]
            avg_q_mag        = np.mean([q[0] for q in recent_q_mags])
            avg_target_q_mag = np.mean([q[1] for q in recent_q_mags])
            print(f"Episode {episode+1}/{total_new_episodes} | Avg Reward: {avg_reward:.4f} | "
                  f"Avg Steps: {avg_length:.1f} | Avg RPS: {avg_rps:.4f} | "
                  f"Success Rate: {success_rate:.1%} | Epsilon: {agent.epsilon:.4f}")
            print(f"  Q-Value Mag: {avg_q_mag:.4f} | Target Q-Value Mag: {avg_target_q_mag:.4f}")
        else:
            print(f"Episode {episode+1}/{total_new_episodes} | Avg Reward: {avg_reward:.4f} | "
                  f"Avg Steps: {avg_length:.1f} | Avg RPS: {avg_rps:.4f} | "
                  f"Success Rate: {success_rate:.1%} | Epsilon: {agent.epsilon:.4f}")

        if (episode + 1) % 10 == 0:
            torch.save(agent.q_network.state_dict(), f"dqn_checkpoint_ep{episode+1}.pt")
            print(f"Checkpoint saved at episode {episode+1}")

    print("\n" + "="*60)
    print("Training completed!")
    print(f"Total Successful Episodes (terminal): {total_success}")
    print(f"Total Truncated Episodes: {total_truncated}")
    print(f"Success Rate: {total_success}/{episode+1} = {total_success/(episode+1):.1%}")
    print("="*60)

    # Update and save stats
    stats['episode_rewards']   = episode_rewards
    stats['episode_lengths']   = episode_lengths
    stats['episode_successes'] = episode_successes
    stats['q_magnitudes']      = q_magnitudes
    stats['losses']            = losses
    stats['total_success']     = total_success
    stats['total_truncated']   = total_truncated
    stats['hyperparameters'] = {
        'GAMMA':                      GAMMA,
        'EPSILON':                    EPSILON,
        'EPSILON_DECAY':              EPSILON_DECAY,
        'EPSILON_DECAY_ITER':         EPSILON_DECAY_ITER,
        'MIN_EPSILON':                MIN_EPSILON,
        'LEARNING_RATE':              LEARNING_RATE,
        'BATCH_SIZE':                 BATCH_SIZE,
        'UPDATE_FREQ':                UPDATE_FREQ,
        'TARGET_NETWORK_UPDATE_FREQ': TARGET_NETWORK_UPDATE_FREQ,
        'BUFFER_LENGTH':              BUFFER_LENGTH,
        'N_EPISODES':                 total_new_episodes,
        'N_ACTIONS':                  N_ACTIONS,
    }

    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)
    print(f"Updated training stats saved as '{stats_path}'")

    torch.save(agent.q_network.state_dict(), "dqn_final_model.pt")
    print("Final model saved as 'dqn_final_model.pt'")

    return agent, episode_rewards, episode_lengths, episode_successes, q_magnitudes, losses


def print_saved_stats(stats):
    """Print saved training statistics."""
    print("\n" + "="*60)
    print("SAVED TRAINING STATISTICS")
    print("="*60)
    print(f"Total Successful Episodes (terminal): {stats['total_success']}")
    print(f"Total Truncated Episodes: {stats['total_truncated']}")
    success_rate = stats['total_success'] / (stats['total_success'] + stats['total_truncated'])
    print(f"Success Rate: {success_rate:.1%}")
    print(f"Mean Episode Reward: {np.mean(stats['episode_rewards']):.4f}")
    print(f"Mean Episode Length: {np.mean(stats['episode_lengths']):.2f}")
    print("\nHyperparameters:")
    for key, value in stats['hyperparameters'].items():
        print(f"  {key}: {value}")
    print("="*60 + "\n")


def plot_training_curves(episode_rewards, episode_lengths, episode_successes, q_magnitudes, losses):
    """Plot training curves and metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    window = 10

    # Plot 1: Episode Rewards
    ax = axes[0, 0]
    ax.plot(episode_rewards, alpha=0.6, label='Episode Reward')
    moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
    ax.plot(range(window-1, len(episode_rewards)), moving_avg, 'r-', linewidth=2, label=f'{window}-Episode MA')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Episode Rewards Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Episode Lengths
    ax = axes[0, 1]
    ax.plot(episode_lengths, alpha=0.6, label='Episode Length')
    moving_avg_len = np.convolve(episode_lengths, np.ones(window)/window, mode='valid')
    ax.plot(range(window-1, len(episode_lengths)), moving_avg_len, 'g-', linewidth=2, label=f'{window}-Episode MA')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps')
    ax.set_title('Episode Lengths Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Success Rate
    ax = axes[1, 0]
    cumulative_success = np.cumsum(episode_successes) / np.arange(1, len(episode_successes) + 1)
    ax.plot(cumulative_success, 'b-', linewidth=2, label='Cumulative Success Rate')
    moving_avg_success = np.convolve(episode_successes, np.ones(window)/window, mode='valid')
    ax.plot(range(window-1, len(episode_successes)), moving_avg_success, 'r--', linewidth=2, label=f'{window}-Episode MA')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Success Rate')
    ax.set_title('Cumulative Success Rate (Terminal Episodes)')
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Q-Value Magnitudes
    ax = axes[1, 1]
    if q_magnitudes:
        q_mags        = [q[0] for q in q_magnitudes]
        target_q_mags = [q[1] for q in q_magnitudes]
        ax.plot(q_mags,        alpha=0.5, label='Q-Value Magnitude')
        ax.plot(target_q_mags, alpha=0.5, label='Target Q-Value Magnitude')
        if len(q_mags) > window:
            moving_avg_q = np.convolve(q_mags, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(q_mags)), moving_avg_q, 'r-', linewidth=2, label='Q-Value MA')
    ax.set_xlabel('Training Update')
    ax.set_ylabel('Magnitude')
    ax.set_title('Q-Value Magnitudes During Training')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('dqn_training_curves.png', dpi=150)
    print("Training curves saved as 'dqn_training_curves.png'")
    plt.show()


def plot_saved_training_curves(stats):
    """Plot training curves from saved stats."""
    plot_training_curves(
        stats['episode_rewards'],
        stats['episode_lengths'],
        stats['episode_successes'],
        stats['q_magnitudes'],
        stats['losses']
    )


def plot_loss_curve(losses):
    """Plot loss curve."""
    plt.figure(figsize=(12, 5))
    plt.plot(losses, alpha=0.6, label='Loss')
    window = 50
    if len(losses) > window:
        moving_avg_loss = np.convolve(losses, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(losses)), moving_avg_loss, 'r-', linewidth=2, label=f'{window}-Update MA')
    plt.xlabel('Training Update')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('dqn_loss_curve.png', dpi=150)
    print("Loss curve saved as 'dqn_loss_curve.png'")
    plt.show()


def evaluate_dqn(agent, n_episodes=10, render=False):
    """Evaluate trained DQN agent."""
    env = Hw2Env(n_actions=N_ACTIONS, render_mode="gui" if render else "offscreen")

    episode_rewards = []
    success_count   = 0

    for episode in range(n_episodes):
        env.reset()
        state = env.high_level_state()
        done  = False
        cumulative_reward = 0.0
        episode_steps     = 0

        while not done and episode_steps < MAX_STEPS:
            action = agent.select_action(state, training=False)
            _, reward, is_terminal, is_truncated = env.step(action)
            next_state = env.high_level_state()
            done = is_terminal or is_truncated

            cumulative_reward += reward
            episode_steps     += 1
            state = next_state

            if is_terminal:
                success_count += 1

        episode_rewards.append(cumulative_reward)
        print(f"Evaluation Episode {episode+1}: Reward={cumulative_reward:.4f}, Steps={episode_steps}")

    success_rate = success_count / n_episodes
    avg_reward   = np.mean(episode_rewards)
    print(f"\nEvaluation Results: Success Rate={success_rate:.2%}, Avg Reward={avg_reward:.4f}")

    return episode_rewards, success_rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DQN Training and Evaluation")
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], default='train',
                        help='Run mode: train or eval (default: train)')
    parser.add_argument('--render', type=str, choices=['gui', 'offscreen'], default='offscreen',
                        help='Render mode: gui or offscreen (default: offscreen)')
    parser.add_argument('--resume', type=str, choices=['yes', 'no', 'auto'], default='auto',
                        help='Resume from checkpoint: yes, no, or auto-detect (default: auto)')
    parser.add_argument('--episodes', type=int, default=500,
                        help='Number of episodes for training (default: 500)')
    parser.add_argument('--eval-episodes', type=int, default=10,
                        help='Number of episodes for evaluation (default: 10)')
    parser.add_argument('--model', type=str, default=None,
                        help='Model checkpoint to load for evaluation (e.g., dqn_checkpoint_ep100.pt)')
    args = parser.parse_args()
    
    # Training mode
    if args.mode == 'train':
        checkpoint_files = [f for f in os.listdir('.') if f.startswith('dqn_checkpoint_ep') and f.endswith('.pt')]
        
        # Determine if resuming
        resume_training = False
        if args.resume == 'auto':
            resume_training = len(checkpoint_files) > 0
        elif args.resume == 'yes':
            resume_training = True
        else:  # 'no'
            resume_training = False
        
        if resume_training and checkpoint_files:
            latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('ep')[1].split('.')[0]))
            print(f"Found checkpoint: {latest_checkpoint}")
            print(f"Resuming training from checkpoint (render_mode={args.render})...")
            agent, rewards, lengths, successes, q_mags, train_losses = continue_training(
                checkpoint_path=latest_checkpoint,
                stats_path='dqn_training_stats.pkl',
                additional_episodes=args.episodes,
                render_mode=args.render
            )
        else:
            # Train from scratch
            print(f"Starting fresh training (render_mode={args.render})...")
            agent, rewards, lengths, successes, q_mags, train_losses = train_dqn()
            torch.save(agent.q_network.state_dict(), "dqn_final_model.pt")
            print("Final model saved as 'dqn_final_model.pt'")
        
        # Plot training curves
        plot_training_curves(rewards, lengths, successes, q_mags, train_losses)
        plot_loss_curve(train_losses)
    
    # Evaluation mode
    elif args.mode == 'eval':
        # Determine which model to load
        if args.model:
            model_path = args.model
            if not os.path.exists(model_path):
                print(f"Model file not found: {model_path}")
                exit(1)
        else:
            # Auto-detect: try final model first, then latest checkpoint
            if os.path.exists('dqn_final_model.pt'):
                model_path = 'dqn_final_model.pt'
            else:
                checkpoint_files = [f for f in os.listdir('.') if f.startswith('dqn_checkpoint_ep') and f.endswith('.pt')]
                if not checkpoint_files:
                    print("No model found. Train first with --mode train")
                    exit(1)
                model_path = max(checkpoint_files, key=lambda x: int(x.split('ep')[1].split('.')[0]))
        
        # Load model
        agent = DQNAgent(n_actions=N_ACTIONS, state_dim=6)
        agent.q_network.load_state_dict(torch.load(model_path, map_location=device))
        agent.target_network.load_state_dict(agent.q_network.state_dict())
        print(f"Loaded model from: {model_path}")
        
        # Evaluate
        print(f"\nEvaluating agent (render_mode={args.render}, episodes={args.eval_episodes})...")
        print("="*50)
        eval_rewards, success_rate = evaluate_dqn(agent, n_episodes=args.eval_episodes, render=(args.render == 'gui'))