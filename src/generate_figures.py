import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.interpolate import griddata, make_interp_spline

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.baselines import get_empirical_dataset
from src.train import prepare_empirical_batches, bsde_empirical_loss
from src.models import DeepBSDE_RoughVol

FIGS_DIR = os.path.join(BASE_DIR, "Figs")

plt.style.use('default')
plt.rcParams['figure.facecolor'] = '#ffffff'
plt.rcParams['axes.facecolor'] = '#ffffff'
plt.rcParams['axes.edgecolor'] = '#000000'
plt.rcParams['axes.labelcolor'] = '#000000'
plt.rcParams['text.color'] = '#000000'
plt.rcParams['xtick.color'] = '#000000'
plt.rcParams['ytick.color'] = '#000000'
plt.rcParams['grid.color'] = '#cccccc'
plt.rcParams['grid.alpha'] = 0.7
plt.rcParams['font.family'] = 'serif' # Best for LaTeX papers
plt.rcParams['font.size'] = 12

# ==========================================
# Data Visualizations
# ==========================================

def plot_spx_vix_correlation():
    vix_path = os.path.join(BASE_DIR, "Data", "VIX_history.csv")
    spx_path = os.path.join(BASE_DIR, "Data", "SPX_history.csv")
    if not (os.path.exists(vix_path) and os.path.exists(spx_path)): return
    
    vix_df = pd.read_csv(vix_path, index_col=0, parse_dates=True)
    spx_df = pd.read_csv(spx_path, index_col=0, parse_dates=True)
    df = spx_df.join(vix_df, rsuffix='_vix').dropna()
    
    fig, ax1 = plt.subplots(figsize=(10, 5), dpi=300)
    ax1.grid(True, linestyle='--', zorder=0)
    
    # SPX Line with fill
    l1, = ax1.plot(df.index, df['SPX'], color='#1f77b4', label='S&P 500 Index', lw=2, zorder=3)
    ax1.fill_between(df.index, df['SPX'], df['SPX'].min()*0.95, color='#1f77b4', alpha=0.1, zorder=2)
    ax1.set_ylabel('S&P 500 Level', color='#1f77b4', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    
    # VIX Line on twin axis
    ax2 = ax1.twinx()
    l2, = ax2.plot(df.index, df['VIX'], color='#d62728', label='VIX Volatility', lw=1.5, alpha=0.85, zorder=4)
    ax2.fill_between(df.index, df['VIX'], 0, color='#d62728', alpha=0.1, zorder=1)
    ax2.set_ylabel('VIX Level', color='#d62728', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#d62728')
    ax2.set_ylim(0, df['VIX'].max()*1.2)
    
    fig.suptitle('Empirical S&P 500 & VIX Regime Dynamics', fontsize=16, fontweight='bold', y=0.95)
    fig.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, "spx_vix_correlation.png"), bbox_inches='tight')
    plt.close()

def plot_iv_surface():
    data_dir = os.path.join(BASE_DIR, "Data")
    files = [f for f in os.listdir(data_dir) if f.startswith("SPX_options_chain")]
    if not files: return
    df = pd.read_csv(os.path.join(data_dir, sorted(files)[-1]))
    
    fig = plt.figure(figsize=(10, 7), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('white')
    
    spot = df['Spot'].iloc[0]
    df_f = df[(df['strike'] > spot*0.7) & (df['strike'] < spot*1.3)]
    s = df_f['strike'].values
    m = df_f['Maturity'].values * 365.25
    iv = df_f['impliedVolatility'].values
    
    grid_s, grid_m = np.mgrid[min(s):max(s):100j, min(m):max(m):100j]
    grid_iv = griddata((s, m), iv, (grid_s, grid_m), method='cubic')
    
    surf = ax.plot_surface(grid_s, grid_m, grid_iv, cmap='viridis', edgecolor='none', rstride=1, cstride=1, alpha=0.9, antialiased=True)
    
    ax.contour(grid_s, grid_m, grid_iv, zdir='z', offset=np.nanmin(grid_iv)-0.1, cmap='viridis', alpha=0.5)
    
    ax.set_xlabel('Strike Price (USD)', labelpad=10)
    ax.set_ylabel('Maturity (Days)', labelpad=10)
    ax.set_zlabel('Implied Volatility (IV)', labelpad=10)
    ax.view_init(elev=25, azim=-125)
    ax.set_title("Interpolated S&P 500 Implied Volatility Surface", fontsize=16, fontweight='bold', pad=20)
    
    cb = fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.1)
    cb.set_label('Implied Volatility')
    plt.savefig(os.path.join(FIGS_DIR, "iv_surface_3d.png"), bbox_inches='tight')
    plt.close()

def plot_volatility_smile():
    data_dir = os.path.join(BASE_DIR, "Data")
    files = [f for f in os.listdir(data_dir) if f.startswith("SPX_options_chain")]
    if not files: return
    df = pd.read_csv(os.path.join(data_dir, sorted(files)[-1]))
    
    target_maturity = df['Maturity'].unique()[len(df['Maturity'].unique())//4]
    slice_df = df[df['Maturity'] == target_maturity].sort_values('strike')
    
    x = slice_df['strike'].values
    y = slice_df['impliedVolatility'].values
    x_new = np.linspace(x.min(), x.max(), 300)
    try:
        spline = make_interp_spline(x, y, k=3)
        y_smooth = spline(x_new)
    except:
        y_smooth = np.interp(x_new, x, y)
    
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.grid(True, linestyle=':', color='#cccccc', zorder=0)
    
    ax.scatter(x, y, color='#2ca02c', s=40, zorder=5, label='Empirical Market Quotes', edgecolor='black', linewidth=0.5)
    l, = ax.plot(x_new, y_smooth, color='#2ca02c', lw=2, zorder=4, label='Interpolated Smile')
    
    spot = df['Spot'].iloc[0]
    ax.axvline(spot, color='#7f7f7f', linestyle='--', lw=1.5, label=f'ATM Spot = {spot:.2f}', zorder=2)
    
    ax.set_title(f'Volatility Smile for Target Maturity: {target_maturity*365.25:.1f} Days', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Option Strike Price (USD)', fontweight='bold')
    ax.set_ylabel('Implied Volatility (IV)', fontweight='bold')
    ax.legend(facecolor='white', edgecolor='black', framealpha=1.0)
    fig.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, "volatility_smile.png"), bbox_inches='tight')
    plt.close()

# ==========================================
# Neural Net Visualizations
# ==========================================

def plot_nn_architecture():
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    ax.axis('off')
    
    box_style = "round,pad=0.6,rounding_size=0.1"
    
    t1 = ax.text(0.1, 0.65, "Market Histories\n[Spot(t), VIX(t)]\nDim: (B, 20, 2)", 
                 ha="center", va="center", size=10, color='black', 
                 bbox=dict(boxstyle=box_style, facecolor="#f8f9fa", edgecolor="#333333", lw=1.5))
    
    t2 = ax.text(0.35, 0.65, "LSTM Encoder\nExtracts Non-Markovian\nRough Features\nDim: (B, Hidden)", 
                 ha="center", va="center", size=11, color='black', fontweight='bold',
                 bbox=dict(boxstyle=box_style, facecolor="#e1bee7", edgecolor="#8e44ad", lw=2))
    
    t3 = ax.text(0.35, 0.3, "Option Contract\n[Time, Strike]\nDim: (B, 2)", 
                 ha="center", va="center", size=10, color='black',
                 bbox=dict(boxstyle=box_style, facecolor="#f8f9fa", edgecolor="#333333", lw=1.5))
                 
    t_cat = ax.text(0.55, 0.475, "Concat Eq: \n[Latent H, T, K]", 
                 ha="center", va="center", size=10, color='black', fontweight='bold',
                 bbox=dict(boxstyle="circle,pad=0.3", facecolor="#b3e5fc", edgecolor="#2980b9", lw=2))
                 
    t_price = ax.text(0.78, 0.65, "Pricer MLP\nDense Layers\nOutput: V(t,S)", 
                 ha="center", va="center", size=11, color='black', fontweight='bold',
                 bbox=dict(boxstyle=box_style, facecolor="#c8e6c9", edgecolor="#27ae60", lw=2))
                 
    t_hedge = ax.text(0.78, 0.3, "Hedging MLP\nDense Layers\nOutput: Delta(t,S)", 
                 ha="center", va="center", size=11, color='black', fontweight='bold',
                 bbox=dict(boxstyle=box_style, facecolor="#ffccbc", edgecolor="#d35400", lw=2))

    arrow_props = dict(arrowstyle="->", color="#333333", lw=2)
    ax.annotate("", xy=(0.22, 0.65), xytext=(0.18, 0.65), arrowprops=arrow_props)
    
    arrow_merge = dict(arrowstyle="->", color="#333333", lw=2, connectionstyle="angle,angleA=0,angleB=90,rad=15")
    ax.annotate("", xy=(0.485, 0.475), xytext=(0.47, 0.65), arrowprops=arrow_merge)
    
    arrow_merge2 = dict(arrowstyle="->", color="#333333", lw=2, connectionstyle="angle,angleA=0,angleB=-90,rad=15")
    ax.annotate("", xy=(0.485, 0.475), xytext=(0.47, 0.3), arrowprops=arrow_merge2)
    
    ax.annotate("", xy=(0.69, 0.65), xytext=(0.605, 0.475), arrowprops=arrow_merge)
    ax.annotate("", xy=(0.69, 0.3), xytext=(0.605, 0.475), arrowprops=arrow_merge2)
    
    ax.annotate("", xy=(0.92, 0.65), xytext=(0.86, 0.65), arrowprops=dict(arrowstyle="->", color="#333333", lw=2.5))
    ax.text(0.93, 0.65, "$Price$", color='black', ha='left', va='center', size=14, fontweight='bold')
    
    ax.annotate("", xy=(0.92, 0.3), xytext=(0.86, 0.3), arrowprops=dict(arrowstyle="->", color="#333333", lw=2.5))
    ax.text(0.93, 0.3, r"$\Delta_t$", color='black', ha='left', va='center', size=16, fontweight='bold')
    
    fig.text(0.5, 0.9, "Deep BSDE Neural Architecture Diagram", 
             ha="center", va="center", fontsize=16, fontweight='bold', color='black')

    plt.savefig(os.path.join(FIGS_DIR, "nn_architecture.png"), bbox_inches='tight')
    plt.close()

def plot_gradient_flow_and_3d_error():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepBSDE_RoughVol().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    try:
        X_paths, X_contract, Y_target, price_scaler, spot_scaler, strike_scaler = prepare_empirical_batches(seq_len=20)
        X_paths, X_contract, Y_target = X_paths.to(device), X_contract.to(device), Y_target.to(device)
    except:
        return
        
    model.train()
    optimizer.zero_grad()
    p, g = model(X_paths, X_contract)
    loss = bsde_empirical_loss(p, Y_target, g)
    loss.backward()
    
    layers = []
    ave_grads = []
    max_grads = []
    for n, q in model.named_parameters():
        if q.requires_grad and "bias" not in n and q.grad is not None:
            layers.append(n.replace('.weight', ''))
            ave_grads.append(q.grad.abs().mean().cpu().item())
            max_grads.append(q.grad.abs().max().cpu().item())
            
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    ax.grid(True, axis='y', linestyle='-', color='#eeeeee', zorder=0)
    
    x_pos = np.arange(len(layers))
    width = 0.35
    
    ax.bar(x_pos - width/2, max_grads, width, color='#3498db', label='Max Gradient', edgecolor='black', linewidth=0.5, zorder=3)
    ax.bar(x_pos + width/2, ave_grads, width, color='#e74c3c', label='Mean Gradient', edgecolor='black', linewidth=0.5, zorder=3)
    
    ax.axhline(0, color='black', lw=1, zorder=4)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(layers, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Absolute Gradient Norm', fontweight='bold')
    ax.set_title('PyTorch Backpropagation Gradient Flow', fontsize=14, fontweight='bold', pad=15)
    
    ax.legend(facecolor='white', edgecolor='black', framealpha=1.0)
    fig.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, "gradient_flow.png"), bbox_inches='tight')
    plt.close()
    
    model_path = os.path.join(BASE_DIR, "Data", "DeepBSDE_empirical.pth")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        
    model.eval()
    with torch.no_grad():
        pred_prices, _ = model(X_paths, X_contract)
        
    pred_raw = price_scaler.inverse_transform(pred_prices.cpu().numpy().reshape(-1,1)).flatten()
    targ_raw = price_scaler.inverse_transform(Y_target.cpu().numpy().reshape(-1,1)).flatten()
    strikes_raw = strike_scaler.inverse_transform(X_contract[:,1].cpu().numpy().reshape(-1,1)).flatten()
    
    errors = pred_raw - targ_raw
    maturities = X_contract[:, 0].cpu().numpy() * 365.25
    
    fig = plt.figure(figsize=(10, 7), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('white')
    
    grid_s, grid_m = np.mgrid[min(strikes_raw):max(strikes_raw):100j, min(maturities):max(maturities):100j]
    grid_err = griddata((strikes_raw, maturities), errors, (grid_s, grid_m), method='cubic', fill_value=0.0)
    
    norm = plt.Normalize(vmin=-np.max(np.abs(grid_err)), vmax=np.max(np.abs(grid_err)))
    surf = ax.plot_surface(grid_s, grid_m, grid_err, cmap='coolwarm', norm=norm, edgecolor='none', rstride=1, cstride=1, alpha=0.9, antialiased=True)
    
    ax.plot_surface(grid_s, grid_m, np.zeros_like(grid_err), color='gray', alpha=0.2)
    
    ax.set_xlabel('Strike Price (USD)', labelpad=10)
    ax.set_ylabel('Maturity (Days)', labelpad=10)
    ax.set_zlabel('Pricing Error (USD)', labelpad=10)
    ax.view_init(elev=30, azim=45)
    ax.set_title("Deep BSDE Network: Empirical Pricing Residuals", fontsize=14, fontweight='bold', pad=20)
    
    cb = fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.1)
    cb.set_label('Residual Error (USD)')
    
    plt.savefig(os.path.join(FIGS_DIR, "loss_surface_training.png"), bbox_inches='tight')
    plt.close()

def plot_training_loss():
    from src.train import train_model
    import io
    from contextlib import redirect_stdout
    
    with io.StringIO() as buf, redirect_stdout(buf):
        model, loss_history, _, _, _ = train_model(epochs=500, lr=1e-3)
        
    epochs = np.arange(1, len(loss_history) + 1)
    
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.grid(True, linestyle=':', color='#cccccc', zorder=0)
    
    try:
        x_smooth = np.linspace(epochs.min(), epochs.max(), 300)
        spline = make_interp_spline(epochs, loss_history, k=3)
        y_smooth = spline(x_smooth)
    except:
        x_smooth = epochs
        y_smooth = loss_history
    
    ax.scatter(epochs, loss_history, color='#34495e', s=10, alpha=0.5, zorder=2)
    ax.plot(x_smooth, y_smooth, color='#2980b9', lw=2.5, zorder=3, label='Empirical Loss')
    
    ax.set_title('Neural Convergence: MSE + Hedging Penalty', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Optimization Epochs', fontweight='bold')
    ax.set_ylabel('Strict Empirical Loss', fontweight='bold')
    ax.set_yscale('log')
    ax.legend(facecolor='white', edgecolor='black')
    
    fig.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, "training_loss.png"), bbox_inches='tight')
    plt.close()

def plot_pricing_surfaces_3d():
    from src.train import prepare_empirical_batches
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepBSDE_RoughVol().to(device)
    model_path = os.path.join(BASE_DIR, "Data", "DeepBSDE_empirical.pth")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    try:
        X_paths, X_contract, Y_target, price_scaler, spot_scaler, strike_scaler = prepare_empirical_batches(seq_len=20)
    except:
        return
        
    fig = plt.figure(figsize=(15, 6), dpi=300)
    fig.patch.set_facecolor('white')
    fig.suptitle('Deep Learning Pricing Surfaces: Output Price Mapping', fontsize=16, fontweight='bold', y=0.95)
    
    ax1 = fig.add_subplot(121, projection='3d')
    spot_min, spot_max = spot_scaler.inverse_transform([[-1.5], [1.5]])
    spots_raw = np.linspace(spot_min[0], spot_max[0], 40)
    times_raw = np.linspace(0.01, 1.0, 40)
    S_mesh, T_mesh = np.meshgrid(spots_raw, times_raw)
    
    base_path_scaled = X_paths[0].clone()
    fixed_strike_scaled = strike_scaler.transform([[spot_min[0] + (spot_max[0]-spot_min[0])/2]])[0,0]
    prices_ZT = np.zeros_like(S_mesh)
    
    for i in range(40):
        for j in range(40):
            s_scl = spot_scaler.transform([[S_mesh[i, j]]])[0,0]
            t_val = T_mesh[i, j]
            p_tensor = base_path_scaled.clone().unsqueeze(0).to(device)
            p_tensor[0, -1, 0] = s_scl 
            
            c_tensor = torch.tensor([[t_val, fixed_strike_scaled]], dtype=torch.float32).to(device)
            
            with torch.no_grad():
                pred, _ = model(p_tensor, c_tensor)
            prices_ZT[i, j] = price_scaler.inverse_transform(pred.cpu().numpy())[0,0]
            
    surf1 = ax1.plot_surface(S_mesh, T_mesh * 365.25, prices_ZT, cmap='viridis', edgecolor='none', alpha=0.9, antialiased=True)
    ax1.set_xlabel('Spot Price (USD)', labelpad=10)
    ax1.set_ylabel('Time to Maturity (Days)', labelpad=10)
    ax1.set_zlabel('Call Price (USD)', labelpad=10)
    ax1.set_title('Spot vs. Time (Fixed Variance)', fontsize=12, fontweight='bold', pad=10)
    
    ax2 = fig.add_subplot(122, projection='3d')
    variances_raw = np.linspace(0.005, 0.05, 40)
    S_mesh_v, V_mesh = np.meshgrid(spots_raw, variances_raw)
    prices_ZV = np.zeros_like(S_mesh_v)
    fixed_time = 0.25
    
    for i in range(40):
        for j in range(40):
            s_scl = spot_scaler.transform([[S_mesh_v[i, j]]])[0,0]
            v_val = V_mesh[i, j]
            p_tensor = base_path_scaled.clone().unsqueeze(0).to(device)
            p_tensor[0, -1, 0] = s_scl
            p_tensor[0, -1, 1] = v_val
            
            c_tensor = torch.tensor([[fixed_time, fixed_strike_scaled]], dtype=torch.float32).to(device)
            
            with torch.no_grad():
                pred, _ = model(p_tensor, c_tensor)
            prices_ZV[i, j] = price_scaler.inverse_transform(pred.cpu().numpy())[0,0]
            
    surf2 = ax2.plot_surface(S_mesh_v, V_mesh, prices_ZV, cmap='plasma', edgecolor='none', alpha=0.9, antialiased=True)
    ax2.set_xlabel('Spot Price (USD)', labelpad=10)
    ax2.set_ylabel('Local Variance (VIX^2)', labelpad=10)
    ax2.set_zlabel('Call Price (USD)', labelpad=10)
    ax2.set_title('Spot vs. Variance (Fixed Time)', fontsize=12, fontweight='bold', pad=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, "pricing_surfaces_3d.png"), bbox_inches='tight')
    plt.close()

# ==========================================
# Validation & Empirical Hedging Implementation
# ==========================================

def plot_performance_benchmarks():
    try:
        metrics = np.load(os.path.join(BASE_DIR, "Data", "inference_metrics.npy"), allow_pickle=True).item()
    except Exception as e:
        print("Missing inference metrics:", e)
        return
        
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    ax.grid(True, axis='y', linestyle=':', color='#cccccc', zorder=0)
    
    names = list(metrics.keys())
    values = list(metrics.values())
    
    throughput = [100000 / v for v in values]
    
    colors = ['#1f77b4', '#d62728']
    bars = ax.bar(names, throughput, color=colors, edgecolor='black', zorder=3, width=0.5)
    
    ax.set_ylabel('Physical Throughput (Contracts / Second)', fontweight='bold')
    ax.set_title('Inference Latency: Neural Weights vs Local Analytical CPUs', fontsize=12, fontweight='bold', pad=15)
    ax.set_yscale('log')
    
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval * 1.5, f'{int(yval):,}/sec', ha='center', va='bottom', fontweight='bold', size=11)
        
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, "performance_benchmarks.png"), bbox_inches='tight')
    plt.close()
    print("Generated: performance_benchmarks.png")

def plot_hedging_pnl():
    try:
         data = np.load(os.path.join(BASE_DIR, "Data", "empirical_hedging_pnl.npy"), allow_pickle=True).item()
    except Exception as e:
         print("Missing empirical hedging data:", e)
         return
         
    days = data['days']
    bs_pnl = data['pnl_black_scholes']
    dl_pnl = data['pnl_deep_bsde']
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.grid(True, linestyle=':', color='#cccccc', zorder=0)
    
    ax.plot(days, bs_pnl, color='#d62728', lw=2, label='Traditional Black-Scholes Delta', zorder=3)
    ax.plot(days, dl_pnl, color='#1f77b4', lw=2.5, label='Deep BSDE Rough Volatility Delta', zorder=4)
    
    ax.fill_between(days, bs_pnl, dl_pnl, where=(np.array(dl_pnl) > np.array(bs_pnl)), color='#1f77b4', alpha=0.15, interpolate=True)
    ax.fill_between(days, bs_pnl, dl_pnl, where=(np.array(dl_pnl) <= np.array(bs_pnl)), color='#d62728', alpha=0.15, interpolate=True)
    
    ax.axhline(0, color='black', lw=1, zorder=2)
    
    ax.set_xlabel('Trading Days Since Inception (2020 COVID-19 Crash Window)', fontweight='bold')
    ax.set_ylabel('Cumulative Hedging P&L (USD per Share)', fontweight='bold')
    ax.set_title('Empirical Daily Rebalanced Portfolio: Neural Deltas vs Linear Analytics', fontsize=14, fontweight='bold', pad=15)
    ax.legend(facecolor='white', edgecolor='black', loc='lower left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, "hedging_pnl_trajectory.png"), bbox_inches='tight')
    plt.close()
    print("Generated: hedging_pnl_trajectory.png")

def plot_institutional_comparison():
    """
    Phase 6: Visual Comparison of Institutional Benchmarks vs Deep BSDE.
    Focuses on the short-term 'Roughness' and Skew geometries.
    """
    print("\n[INSTITUTIONAL PLOTS] Generating rBergomi vs Deep BSDE comparison...")
    
    # 1. Load rBergomi Data
    try:
        S_rB = np.load(os.path.join(BASE_DIR, "Data", "benchmark_rbergomi_S.npy"))
        V_rB = np.load(os.path.join(BASE_DIR, "Data", "benchmark_rbergomi_V.npy"))
    except:
        print("Missing rBergomi benchmark data.")
        return

    # 2. Comparison of Price Trajectories (Roughness)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), dpi=300, sharex=True)
    
    t = np.linspace(0, 1.0, S_rB.shape[1])
    
    # rBergomi Jagginess
    ax1.plot(t, S_rB[0], color='#8e44ad', lw=1.5, label='Rough Bergomi Path (H=0.1)')
    ax1.set_ylabel('Institutional S_t', color='#8e44ad', fontweight='bold')
    ax1.set_title('Non-Markovian Jagginess: rBergomi (SOTA) vs Traditional Models', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Variance Roughness
    ax2.plot(t, V_rB[0], color='#e67e22', lw=1.5, label='Rough Volatility Variance')
    ax2.set_ylabel('Variance V_t', color='#e67e22', fontweight='bold')
    ax2.set_xlabel('Time (Years)', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    fig.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, "roughness_comparison.png"), bbox_inches='tight')
    plt.close()
    
    # 3. SABR Skew Comparison
    try:
        df_sabr = pd.read_csv(os.path.join(BASE_DIR, "Data", "benchmark_sabr.csv"))
        T_target = df_sabr['T'].iloc[0]
        slice_sabr = df_sabr[df_sabr['T'] == T_target]
        
        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        ax.plot(slice_sabr['K'], slice_sabr['IV_SABR'], 'o--', color='#2c3e50', label='SABR (Hagan Approx) - Banking Standard')
        ax.set_title(f'Institutional Benchmark: SABR Volatility Skew (T={T_target})', fontweight='bold')
        ax.set_xlabel('Strike', fontweight='bold')
        ax.set_ylabel('Implied Volatility', fontweight='bold')
        ax.legend()
        ax.grid(True, linestyle=':')
        
        plt.savefig(os.path.join(FIGS_DIR, "sabr_skew_benchmark.png"), bbox_inches='tight')
        plt.close()
    except:
        pass
    
    print("Generated: Institutional comparison figures.")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    plot_spx_vix_correlation()
    plot_iv_surface()
    plot_volatility_smile()
    plot_nn_architecture()
    plot_gradient_flow_and_3d_error()
    plot_training_loss()
    plot_pricing_surfaces_3d()
    plot_performance_benchmarks()
    plot_hedging_pnl()
    plot_institutional_comparison()
