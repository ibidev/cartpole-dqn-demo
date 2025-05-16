import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

def build_model(state_size, action_size):
    model = Sequential()
    model.add(Dense(24, input_shape=(state_size,), activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(action_size, activation='linear'))  # Linear because we want raw Q-values
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    return model

def train_dqn(episodes=50):
    env = gym.make("CartPole-v1")  # Updated from v0 to v1
    state_size = env.observation_space.shape[0]  # Should be 4
    action_size = env.action_space.n  # Should be 2 (left or right)
    model = build_model(state_size, action_size)

    for episode in range(episodes):
        state, _ = env.reset()  # Unpack state and ignore 'info'
        state = np.array(state).reshape(1, state_size)
        done = False
        total_reward = 0

        while not done:
            if np.random.rand() < 0.1:
                action = np.random.choice(action_size)  # Random action: explore
            else:
                qs = model.predict(state, verbose=0)  # Predict Q-values
                action = np.argmax(qs[0])  # Choose best action: exploit

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = np.array(next_state).reshape(1, state_size)

            target = reward
            if not done:
                future_q = np.max(model.predict(next_state, verbose=0)[0])
                target += 0.9 * future_q  # Discount future rewards

            ts = model.predict(state, verbose=0)
            ts[0][action] = target

            model.fit(state, ts, verbose=0)
            state = next_state
            total_reward += reward

        print(f"Episode {episode + 1}/{episodes} - Total Reward: {total_reward}")
        
    # Save model after training
    model.save("dqn_cartpole.h5")

if __name__ == "__main__":
    train_dqn()
