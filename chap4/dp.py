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

def argmax(d):
    max_value = max(d.values())
    max_key = 0
    for key, value in d.items():
        if value == max_value:
            max_key = key

    return max_key

def greedy_policy(V, env, gamma):
    pi = {}

    for state in env.states():
        action_values = {}

        for action in env.actions():
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            value = r + gamma * V[next_state]
            action_values[action] = value

        max_action = argmax(action_values)
        action_probs = {0:0, 1:0, 2:0, 3:0}
        action_probs[max_action] = 1.0
        pi[state] = action_probs

    return pi

def policy_iter(env, gamma, threshold=0.001, is_render=False):
    pi = defaultdict(lambda: {0:0.25, 1:0.25, 2:0.25, 3:0.25})
    V = defaultdict(lambda:0)

    while True:
        V = policy_eval(pi, V, env, gamma, threshold)
        new_pi = greedy_policy(V, env, gamma)

        if is_render:
            env.render_v(V, pi)

        if new_pi == pi:
            break
        pi = new_pi

    return pi

def value_iter_onestep(V, env, gamma):
    for state in env.states():
        if state == env.goal_state:
            V[state] = 0
            continue

        action_values = []
        for action in env.actions():
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            value = r + gamma * V[next_state]
            action_values.append(value)

        V[state] = max(action_values)
    return V


def value_iter(V, env, gamma, threshold=0.001, is_render=True):
    while True:
        if is_render:
            env.render_v(V)

        old_V = V.copy()
        V = value_iter_onestep(V, env ,gamma)

        delta = 0
        for state in V.keys():
            t = abs(V[state] - old_V[state])
            if delta < t:
                delta = t

            if delta < threshold:
                break
    return V


def main():
    gamma = 0.9
    env = GridWorld()
    V = defaultdict(lambda: 0)
    pi = policy_iter(env, gamma)

#    V = policy_eval(pi, V, env, gamma)
    V = value_iter(V, env, gamma)

    pi=greedy_policy(V, env, gamma)
    env.render_v(V, pi)


if __name__ == '__main__':
    main()
