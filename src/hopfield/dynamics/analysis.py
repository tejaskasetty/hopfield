import numpy as np

def monte_carlo_simulations(net, patterns, max_iterations):
    """
    Perform Monte Carlo simulations to estimate the depth and width of the basins of attraction.
    
    Args:
    - net: Hopfield network object.
    - patterns: list of patterns to test.
    - max_iterations: maximum number of iterations to test for each pattern.
    
    Returns:
    - depth: average number of iterations required to converge to a stable state.
    - width: standard deviation of the number of iterations required to converge to a stable state.
    """
    iterations = []
    for pattern in patterns:
        net.set_state(pattern)
        for i in range(max_iterations):
            net.update()
            if net.is_stable():
                iterations.append(i)
                break
    depth = np.mean(iterations)
    width = np.std(iterations)
    return depth, width


def perturbation_analysis(net, pattern, max_perturbation, max_iterations):
    """
    Perform perturbation analysis to estimate the depth and width of the basins of attraction.
    
    Args:
    - net: Hopfield network object.
    - pattern: pattern to test.
    - max_perturbation: maximum amount of perturbation to add to the pattern.
    - max_iterations: maximum number of iterations to test for each perturbation level.
    
    Returns:
    - depths: array of average number of iterations required to converge to a stable state for each perturbation level.
    - widths: array of standard deviations of the number of iterations required to converge to a stable state for each perturbation level.
    """
    perturbations = np.linspace(0, max_perturbation, 10)
    depths = np.zeros_like(perturbations)
    widths = np.zeros_like(perturbations)
    for i, perturbation in enumerate(perturbations):
        net.set_state(pattern + np.random.normal(scale=perturbation, size=pattern.shape))
        iterations = []
        for j in range(max_iterations):
            net.update()
            if net.is_stable():
                iterations.append(j)
                break
        depths[i] = np.mean(iterations)
        widths[i] = np.std(iterations)
    return depths, widths


def lyapunov_analysis(net, pattern, max_iterations):
    """
    Perform Lyapunov analysis to estimate the depth and width of the basins of attraction.
    
    Args:
    - net: Hopfield network object.
    - pattern: pattern to test.
    - max_iterations: maximum number of iterations to test for each perturbation level.
    
    Returns:
    - depth: inverse of the Lyapunov exponent.
    - width: standard deviation of the distance between trajectories that start from different initial conditions.
    """
    trajectory = net.iterate(pattern, max_iterations)
    perturbation = np.random.normal(scale=1e-6, size=pattern.shape)
    perturbed_trajectory = net.iterate(pattern + perturbation, max_iterations)
    distance = np.linalg.norm(trajectory - perturbed_trajectory, axis=1)
    lyapunov_exponent = np.mean(np.log(distance[1:] / distance[:-1]))
    depth = 1 / lyapunov_exponent
    width = np.std(distance)
    return depth, width
