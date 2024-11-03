import random
import gym
import numpy as np
import matplotlib.pyplot as plt

def update_q_table(Q_table, state, action, reward, upcoming_state, rate, discount):
    best_future_q = np.max(Q_table[upcoming_state])      
    Q_table[state, action] = Q_table[state, action] + rate * (reward + discount * best_future_q - Q_table[state, action])
    return Q_table

def epsilon_greedy(Q_table, state, exploring_rate):
  
    if random.uniform(0, 1) < exploring_rate:  
        return env.action_space.sample() 
    else:  
        return np.argmax(Q_table[state])  

if __name__ == "__main__":
    env = gym.make("Taxi-v3", render_mode="human")
    Q_table = np.zeros([env.observation_space.n, env.action_space.n])

    rate = 0.1   
    discount = 0.6  
    exploring_rate = 0.2  
    n_epochs = 1000  
    max_itr_per_epoch = 100 
    list_rewards = []

    for episode in range(n_epochs):
        e_reward = 0
        state, _ = env.reset()  
        for step in range(max_itr_per_epoch):
            action = epsilon_greedy(Q_table, state, exploring_rate)
            upcoming_state, reward, done, _, info = env.step(action)
            Q_table = update_q_table(Q_table, state, action, reward, upcoming_state, rate, discount)

            e_reward += reward
            state = upcoming_state  

            if done:
                break
        
        list_rewards.append(e_reward)
        exploring_rate = max(0.01, exploring_rate * 0.99)


        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}: Total Reward = {e_reward}")

    print("Average reward :", np.mean(list_rewards))

    plt.plot(list_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total of rewards")
    plt.title("Suivi")
    plt.show()

    env.close()
