import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')

plt.rcParams.update({
    "text.usetex": True,      # Enable LaTeX text rendering
    # "font.family": "serif",   # Use serif font
    "font.serif": ["Computer Modern Roman"],  # LaTeX default font
    "font.size": 12,          # Base font size
    # Other font and size settings...
})


def max_eigenvalue_search_test3(epsilon, m_k, n):
    """Estimate the lower and upper bounds of the maximum eigenvalue."""
    lower = max(0, 1 / n)  # Theoretical minimum
    upper = 1.0  # Theoretical upper bound

    for j in range(len(m_k)):
        # Basic lower bound: M >= (m_j - ε)^{1/(j+1)}
        if m_k[j] - epsilon > 0:
            lower = max(lower, max(0, m_k[j] - epsilon) ** (1 / (j + 1)))

        # Basic upper bound: M <= (m_j + ε)^{1/(j+2)}
        upper = min(upper, (m_k[j] + epsilon) ** (1 / (j + 2)))

        # Moment ratio lower bound: M >= (m_{j+1} - ε)/(m_j + ε)
        if j < len(m_k) - 1:
            if m_k[j] + epsilon > 0 and m_k[j + 1] - epsilon > 0:
                ratio_lb = (m_k[j + 1] - epsilon) / (m_k[j] + epsilon)
                lower = max(lower, ratio_lb)

    return lower, min(upper, 1.0)


def run_simulation(eps_list, n_list, degree=4, iterations=1000):
    """Run simulation experiments to collect interval length data for different n and epsilon."""
    # Data structure to store results
    results = defaultdict(list)

    for epsilon in eps_list:
        for n in n_list:
            interval_lengths = []

            for _ in range(iterations):
                # Generate random vector and normalize
                diagonal = np.random.exponential(scale=1.0, size=n)

                # Normalize to get Dirichlet(1, ..., 1)
                diagonal = diagonal / np.sum(diagonal)

                # Calculate moments (from 2nd to degree-th order)
                moments = []
                for j in range(2, degree + 1):
                    error = np.random.normal(scale=np.sqrt(epsilon)/2)
                    if np.abs(error) > epsilon:
                        error = np.sign(error) * epsilon
                    moments.append(np.sum(diagonal ** j) + error)
                # Estimate interval
                lower, upper = max_eigenvalue_search_test3(epsilon, np.array(moments), n)
                interval_lengths.append(upper - lower)

            # Store statistical information
            results['epsilon'].append(epsilon)
            results['n'].append(n)
            results['mean_length'].append(np.mean(interval_lengths))
            results['max_length'].append(np.max(interval_lengths))
            results['min_length'].append(np.min(interval_lengths))
            results['std_dev'].append(np.std(interval_lengths))

    return pd.DataFrame(results)


def plot_results(df):
    """Visualize interval length results for different n and epsilon."""
    plt.figure(figsize=(8, 6))  # Increase figure size

    # Set global font size
    plt.rcParams.update({'font.size': 20})

    markers = ['o', 's', 'D', '^', 'v']

    # Add spacing for different n
    n_values = df['n'].unique()
    n_count = len(n_values)

    # 2. Max interval length vs epsilon (different n)
    for ii, n in enumerate(n_values):
        subset = df[df['n'] == n]

        # Calculate error bars
        y_err = [
            subset['mean_length'] - subset['min_length'],
            subset['max_length'] - subset['mean_length']
        ]

        # Add slight offset for each n to create spacing effect
        x_offset = 0.15 * (ii - n_count / 2)  # Calculate offset

        plt.errorbar(subset['epsilon'] * (1 + x_offset), subset['mean_length'], yerr=y_err,
                     fmt='s-', label=fr'$n={n}$', capsize=6, marker=markers[ii],
                     markersize=8, linewidth=2.5, elinewidth=2)

    plt.xlabel(r'Moment error bound $\varepsilon$', fontsize=28)
    plt.ylabel('Interval Length', fontsize=28)
    plt.title(r'Interval Length vs $\varepsilon$', fontsize=36)
    plt.xscale('log')
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)

    # Increase legend font size
    legend = plt.legend(title='System size', fontsize=20, title_fontsize=22, ncol=2)

    plt.grid(True, which='both', linestyle='--')
    plt.tight_layout()  # Automatically adjust layout
    plt.savefig('interval-length.pdf', bbox_inches='tight')  # Ensure all elements are included when saving
    plt.show()


if __name__ == "__main__":
    # Parameter settings
    eps_list = [0.01, 0.001, 1e-4, 1e-5, 0]
    n_list = [2, 4, 8, 16, 32]  # Increased n to observe larger vectors
    iterations = 1000  # Reduce iterations to speed up

    # Run simulation
    results_df = run_simulation(eps_list, n_list, degree=4, iterations=iterations)

    # Print simulation results summary
    print("Simulation Results Summary:")
    print(results_df.groupby(['n', 'epsilon']).agg({
        'mean_length': 'mean',
        'max_length': 'max',
        'min_length': 'min',
        'std_dev': 'mean'
    }))

    # Visualize results
    plot_results(results_df)
