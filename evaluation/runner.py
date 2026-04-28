from evaluation.backtest import run_strategy
from evaluation.plotting import plot_results

def compare(env, agents: dict):
    results = {}

    for name, agent in agents.items():
        results[name] = run_strategy(env, agent)

    plot_results(results)

    return results