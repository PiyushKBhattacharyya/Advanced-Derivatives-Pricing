import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def plot_historical_roughness(t, paths, title="Empirical Historical Paths"):
    """
    Plots historical index paths to visualize the realized empirical roughness of the trajectories.
    
    Args:
        t (np.ndarray): Time vector in years.
        paths (np.ndarray): Historical realized asset or volatility paths.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(10, 6))
    for i in range(min(5, paths.shape[0])): 
        plt.plot(t, paths[i], label=f"Path {i+1}", alpha=0.8)
    
    plt.title(f"{title}")
    plt.xlabel("Time (Years)")
    plt.ylabel("Index Value")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_3d_volatility_surface(maturities, strikes, ivs, title="3D Empirical Volatility Surface"):
    """
    Renders an interactive 3D Volatility Surface utilizing Plotly.
    Maps empirical Options Chain data (Maturity, Strike) to Implied Volatility.
    
    Args:
        maturities (np.ndarray): Vector of maturities in years.
        strikes (np.ndarray): Vector of strike prices.
        ivs (np.ndarray): 2D Grid mapping of Implied Volatility percentages.
        title (str): Output rendering title.
    """
    fig = go.Figure(data=[go.Surface(
        x=maturities, 
        y=strikes, 
        z=ivs,
        colorscale='Viridis'
    )])
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='Maturity (Years)',
            yaxis_title='Strike Price',
            zaxis_title='Implied Volatility (IV)',
            camera=dict(
                eye=dict(x=1.5, y=-1.5, z=1.2)
            )
        ),
        autosize=False,
        width=800,
        height=800,
        margin=dict(l=65, r=50, b=65, t=90)
    )
    
    fig.show()

def plot_3d_pricing_error_surface(spots, maturities, errors, title="3D Empirical Pricing Error"):
    """
    Visualizes the residuals of the neural network pricer against the actual market ground truth.
    
    Args:
        spots (np.ndarray): Grid of historical spot prices.
        maturities (np.ndarray): Grid of time to maturities.
        errors (np.ndarray): Pre-calculated pricing residuals (Model - Market).
        title (str): Output rendering title.
    """
    fig = go.Figure(data=[go.Surface(
        x=spots, 
        y=maturities, 
        z=errors,
        colorscale='RdBu', 
        cmin=-np.max(np.abs(errors)),
        cmax=np.max(np.abs(errors))
    )])
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='Spot Price',
            yaxis_title='Time to Maturity',
            zaxis_title='Residual Pricing Error',
        ),
        width=800,
        height=800
    )
    
    fig.show()
