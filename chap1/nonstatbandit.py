import numpy as np
import matplotlib.pyplot as plt

runs = 200
steps = 1000
epsilon = 0.1
all_rates = np.zeros((runs, steps))


class Bandit:
    def __init__(self, arms=10):
        self.rates = np.random.rand(arms)

    def play(self, arm):
        rate = self.rates[arm]
        if rate > np.random.rand():
            return 1
        else:
            return 0


class NonStatBandit:
    def __init__(self, arms=10):
        self.rates = np.random.rand(arms)
        self.arms = arms

    def play(self, arm):
        rate = self.rates[arm]
        self.rates += 0.1 * np.random.rand(self.arms)
        if rate > np.random.rand():
            return 1
        else:
            return 0


class Agent:
    def __init__(self, epsilon, action_size=10):
        self.epsilon = epsilon
        self.Qs = np.zeros(action_size)
        self.ns = np.zeros(action_size)

    def update(self, action, reward):
        self.ns[action] += 1
        self.Qs[action] += (reward - self.Qs[action]) / self.ns[action]

    def get_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, len(self.Qs))
        return np.argmax(self.Qs)


class AlphaAgent:
    def __init__(self, epsilon, alpha, action_size=10):
        self.epsilon = epsilon
        self.Qs = np.zeros(action_size)
        self.alpha = alpha

    def update(self, action, reward):
        self.Qs[action] += (reward - self.Qs[action]) * self.alpha

    def get_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, len(self.Qs))
        return np.argmax(self.Qs)


def main():
    for run in range(runs):
        bandit = NonStatBandit()
        #agent = Agent(epsilon)
        agent = AlphaAgent(epsilon, 0.8)
        total_reward = 0
        rates = []

        for step in range(steps):
            action = agent.get_action()
            reward = bandit.play(action)
            agent.update(action, reward)
            total_reward += reward

            rates.append(total_reward / (step+1))

        all_rates[run] = rates

    avg_rates = np.average(all_rates, axis=0)


    plt.ylabel('Rates')
    plt.xlabel('Steps')
    plt.ylim(0.8, 1.0)
    plt.plot(avg_rates)
    plt.show()



if __name__ == '__main__':
    main()
