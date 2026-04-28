import matplotlib.pyplot as plt
from evaluation.metrics import compute_drawdown

def plot_results(results: dict):

    plt.figure(figsize=(12, 6))

    for name, nav in results.items():
        plt.plot(nav, label=name)

    plt.title("Equity Curves")
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(12, 4))

    for name, nav in results.items():
        dd = compute_drawdown(nav)
        plt.plot(dd, label=name)

    plt.title("Drawdowns")
    plt.legend()
    plt.grid()
    plt.show()