import matplotlib.pyplot as plt


def visualization(returns, var, cvar):
    plt.hist(returns)
    plt.axvline(var, label='VaR')
    plt.axvline(cvar, linestyle='dashed', label='cVaR')
    plt.axvspan(xmin=returns.min().min(), xmax=var, color='red', alpha=0.3)
    plt.title('Historical VaR and cVaR (ES)')
    plt.xlabel('Returns')
    plt.ylabel('Frequency')

    return plt.legend(), plt.show()
