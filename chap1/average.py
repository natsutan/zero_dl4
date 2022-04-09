import numpy as np

np.random.seed(0)


def ave0(num):
    rewards = []
    for n in range(1, num+1):
        reward = np.random.rand()
        rewards.append(reward)
        Q = sum(rewards) / n
        print(Q)

def ave1(num):
    rewards = []
    Q = 0
    for n in range(1, num+1):
        reward = np.random.rand()
        Q = Q + (reward - Q) / n
        print(Q)

def main():
    ave1(10)

if __name__ == '__main__':
    main()
