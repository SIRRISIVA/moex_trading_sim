import numpy as np

def run_strategy(env, agent):
    obs = env.reset()
    done = False

    navs = []

    while not done:
        action = agent.act(obs)

        obs, reward, done, info = env.step(action)
        navs.append(info["nav"])

    return np.array(navs)