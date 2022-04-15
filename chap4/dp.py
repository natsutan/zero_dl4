import sys
from collections import defaultdict
import numpy as np

sys.path.append("../common")
from World import GridWorld


def eval_onestep(pi, V, env, gamma=0.9):
    for state in env.states():
        if state == env.goal_state:
            V[state] = 0
            continue

        acton_probs = pi[state]
        new_V = 0

        for action, acton_probs in acton_probs.items():
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            new_V += acton_probs * (r + gamma * V[next_state])

        V[state] = new_V

    return V

def policy_eval(pi, V, env, gamma, threshold = 0.001):
    while True:
        old_V = V.copy()
        V = eval_onestep(pi, V, env, gamma)

        delta = 0
        for state in V.keys():
            t = abs(V[state] - old_V[state])
            if delta < t:
                delta = t

        if delta  < threshold:
            break

    return V


def main():
    gamma = 0.9
    env = GridWorld()
    V = defaultdict(lambda: 0)
    pi = defaultdict(lambda: {0: 0.25, 1:0.25, 2:0.25, 3:0.25})

    V = policy_eval(pi, V, env, gamma)

    env.render_v(V, pi)


if __name__ == '__main__':
    main()
