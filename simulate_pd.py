"""Simulation of asset values and Merton-implied default probabilities for Section 5.
Writes pd.npz (default probabilities of the three specifications and the realised
asset parameters).  Seeds are fixed inside run_gbm_simulation (numpy seed 2222)."""
import numpy as np
from scipy.stats import norm, random_correlation


def generate_correlation_matrix(dim, correlation_type='random', base_corr=0.3):
    """
    Generate correlation matrix for multidimensional GBM
    
    Parameters:
    dim: Number of assets
    correlation_type: 'random', 'uniform', 'block', or 'exponential'
    base_corr: Base correlation level
    
    Returns:
    corr: Correlation matrix
    """
    if dim == 1:
        return np.array([[1.0]])
    
    if correlation_type == 'uniform':
        # Uniform correlation structure
        corr = np.full((dim, dim), base_corr)
        np.fill_diagonal(corr, 1.0)
        
    elif correlation_type == 'block':
        # Block correlation structure
        corr = np.eye(dim)
        block_size = max(2, dim // 3)
        for i in range(0, dim, block_size):
            end_i = min(i + block_size, dim)
            for j in range(i, end_i):
                for k in range(i, end_i):
                    if j != k:
                        corr[j, k] = base_corr
                        
    elif correlation_type == 'exponential':
        # Exponential decay correlation
        corr = np.eye(dim)
        for i in range(dim):
            for j in range(dim):
                if i != j:
                    corr[i, j] = base_corr ** abs(i - j)
                    
    else:  # random correlation
        # Generate random correlation matrix with specified eigenvalues
        eig = np.random.uniform(0.1, 1, dim)
        eig = eig / np.sum(eig) * dim  # Normalize eigenvalues
        corr = random_correlation.rvs(eig, random_state=2222)
    
    return corr

def GBMsimulator_correlated(seed, S0, mu, sigma, corr_matrix, T, N, 
                          lam=0, m=0, v=0, model_type='standard'):
    """
    Multidimensional Correlated Geometric Brownian Motion simulator
    
    Parameters:
    seed: Random seed for reproducibility
    S0: Initial asset prices (array)
    mu: Drift rates (array)
    sigma: Volatilities (array)
    corr_matrix: Correlation matrix
    T: Time horizon
    N: Number of time steps
    lam: Jump intensity (for jump models)
    m: Jump mean (for jump models)
    v: Jump std deviation (for jump models)
    model_type: 'standard', 'jump', or 'sv_jump'
    
    Returns:
    S: Asset price paths [n_assets x n_steps]
    t: Time grid
    """
    np.random.seed(seed)
    dim = len(S0)
    dt = T / N
    t = np.linspace(0., T, int(N))
    
    # Create covariance matrix from correlation matrix and volatilities
    Cov = np.outer(sigma, sigma) * corr_matrix
    
    # Cholesky decomposition for correlated random numbers
    try:
        A = np.linalg.cholesky(Cov)
    except np.linalg.LinAlgError:
        # If Cholesky fails, use eigenvalue decomposition
        eigenvals, eigenvecs = np.linalg.eigh(Cov)
        eigenvals = np.maximum(eigenvals, 1e-8)  # Ensure positive eigenvalues
        A = eigenvecs @ np.diag(np.sqrt(eigenvals))
    
    # Initialize price paths
    S = np.zeros([dim, int(N)])
    S[:, 0] = S0
    
    # Generate correlated paths
    for i in range(1, int(N)):
        # Generate independent standard normal random variables
        Z = np.random.normal(0., 1., dim)
        
        # Transform to correlated normal variables
        corr_Z = A @ Z
        
        # Calculate drift term
        if model_type == 'standard':
            drift = (mu - 0.5 * sigma**2) * dt
            jump_term = np.zeros(dim)
            
        elif model_type == 'jump':
            drift = (mu - 0.5 * sigma**2 - lam * m) * dt
            # Add Poisson jumps
            jump_counts = np.random.poisson(lam * dt, size=dim)
            jump_sizes = np.random.normal(m, v, size=dim)
            jump_term = jump_counts * jump_sizes
            
        elif model_type == 'sv_jump':
            # Simplified stochastic volatility with jumps
            # (Full Heston model would require additional state variables)
            vol_factor = 1 + 0.1 * np.random.normal(0, 1, dim)  # Simple vol randomness
            effective_sigma = sigma * vol_factor
            drift = (mu - 0.5 * effective_sigma**2 - lam * m) * dt
            corr_Z = corr_Z * vol_factor  # Apply stochastic volatility
            
            # Add jumps
            jump_counts = np.random.poisson(lam * dt, size=dim)
            jump_sizes = np.random.normal(m, v, size=dim)
            jump_term = jump_counts * jump_sizes
        
        # Calculate diffusion term
        diffusion = corr_Z * np.sqrt(dt)
        
        # Update asset prices using geometric Brownian motion
        S[:, i] = S[:, i-1] * np.exp(drift + diffusion + jump_term)
    
    return S, t

def merton_default_probability(asset_value, debt_level, r=0.05, sigma=0.2, T=1.0):
    """
    Calculate default probability using Merton model
    
    Parameters:
    asset_value: Current asset value
    debt_level: Debt threshold
    r: Risk-free rate
    sigma: Asset volatility
    T: Time to maturity
    
    Returns:
    Default probability
    """
    if asset_value <= 0:
        return 1.0
    
    d_minus = (np.log(asset_value / debt_level) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(-d_minus)

def run_gbm_simulation(n_simulations=10000, n_assets=10, correlation_type='uniform', 
                      base_corr=0.3, model_types=['standard', 'jump']):
    """
    Run multidimensional correlated GBM simulation
    
    Parameters:
    n_simulations: Number of Monte Carlo simulations
    n_assets: Number of assets
    correlation_type: Type of correlation structure
    base_corr: Base correlation level
    model_types: List of model types to compare
    
    Returns:
    results: Dictionary containing simulation results
    """
    print(f"Running {n_simulations} simulations of {n_assets}-dimensional correlated GBM...")
    
    # Set random seed for reproducibility
    np.random.seed(2222)
    
    # Simulation parameters
    T = 1.0  # 1 year
    N = 252  # Daily steps
    r = 0.05  # Risk-free rate
    debt_level = 100  # Default threshold
    
    # Asset parameters - more realistic ranges
    S0 = np.random.uniform(80, 120, n_assets)  # Initial prices
    mu = np.random.uniform(0.08, 0.12, n_assets)  # Expected returns
    sigma = np.random.uniform(0.15, 0.30, n_assets)  # Volatilities
    
    # Generate correlation matrix
    corr_matrix = generate_correlation_matrix(n_assets, correlation_type, base_corr)
    
    print(f"Correlation matrix ({correlation_type}):")
    print(f"Average off-diagonal correlation: {(corr_matrix.sum() - n_assets) / (n_assets * (n_assets - 1)):.3f}")
    
    # Storage for results
    results = {}
    
    for model_type in model_types:
        print(f"\nSimulating {model_type} model...")
        
        # Model-specific parameters
        if model_type == 'jump' or model_type == 'sv_jump':
            lam = 2.0    # Jump intensity (2 jumps per year on average)
            m = -0.05    # Negative jump mean (market crashes)
            v = 0.15     # Jump volatility
        else:
            lam = m = v = 0
        
        # Storage for this model
        final_prices = []
        default_probs = []
        price_paths = []
        
        # Run simulations
        for sim in range(n_simulations):
            if sim % 1000 == 0:
                print(f"  Simulation {sim}/{n_simulations}")
            
            # Generate unique seed for each simulation
            sim_seed = np.random.randint(1, 2**31-1)
            
            # Run GBM simulation
            prices, time_grid = GBMsimulator_correlated(
                sim_seed, S0, mu, sigma, corr_matrix, T, N,
                lam, m, v, model_type
            )
            
            # Store final prices
            final_prices.append(prices[:, -1])
            
            # Calculate default probabilities
            sim_default_probs = []
            for asset in range(n_assets):
                prob = merton_default_probability(
                    prices[asset, -1], debt_level, r, sigma[asset], T
                )
                sim_default_probs.append(prob)
            
            default_probs.append(sim_default_probs)
            
            # Store some price paths for visualization (first 100 simulations)
            if sim < 100:
                price_paths.append(prices)
        
        # Store results for this model
        results[model_type] = {
            'final_prices': np.array(final_prices),
            'default_probs': np.array(default_probs),
            'price_paths': np.array(price_paths) if price_paths else None,
            'time_grid': time_grid,
            'parameters': {
                'S0': S0, 'mu': mu, 'sigma': sigma, 
                'lam': lam, 'm': m, 'v': v,
                'correlation_matrix': corr_matrix
            }
        }
    
    return results

if __name__ == "__main__":
    res = run_gbm_simulation(n_simulations=5000, n_assets=10, correlation_type="uniform",
                             base_corr=0.3, model_types=["standard", "jump", "sv_jump"])
    par = res["standard"]["parameters"]
    np.savez("pd.npz", standard=res["standard"]["default_probs"], jump=res["jump"]["default_probs"],
             sv_jump=res["sv_jump"]["default_probs"], S0=par["S0"], mu=par["mu"], sigma=par["sigma"])
