import matplotlib.pyplot as plt


def visualization(returns, var, cvar):
    plt.style.use('seaborn-v0_8')
    plt.hist(returns, bins=50)
    plt.axvline(var, color='red', label=f'VaR (95%) {var:.4f}')
    plt.axvline(cvar, color='green', linestyle='dashed', label=f'cVaR (95%) {cvar:.4f}')
    plt.axvspan(xmin=cvar, xmax=var, color='red', alpha=0.3)
    plt.title(f'Historical VaR and cVaR (ES) for {returns.name}')
    plt.xlabel('Returns')
    plt.ylabel('Frequency')
    plt.xlim(-0.12, 0.10)
    plt.legend()
    plt.show()
