import matplotlib.pyplot as plt


def plot_var_cvar(returns, var, cvar, confidence_level=0.95, xlim=None):
    plt.style.use('seaborn-v0_8')
    plt.hist(returns, bins=50)
    plt.axvline(-var, color='red', label=f'VaR ({confidence_level:.0%}): {var:.2%}')
    plt.axvline(-cvar, color='green', linestyle='dashed', label=f'cVaR ({confidence_level:.0%}): {cvar:.2%}')
    plt.axvspan(xmin=-cvar, xmax=-var, color='red', alpha=0.3)
    plt.title(f'Historical VaR and cVaR (ES) for {returns.name}')
    plt.xlabel('Returns')
    plt.ylabel('Frequency')
    if xlim is not None:
        plt.xlim(xlim)
    plt.legend()
    plt.show()


