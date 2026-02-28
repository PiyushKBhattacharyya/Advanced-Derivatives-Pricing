import torch
import numpy as np
import os
import sys

# Ensure base directory is in path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.models import DeepBSDE_RoughVol
from src.train import prepare_empirical_batches

class DeepExplainer:
    """
    Implements Gradient-based Feature Attribution for the Deep BSDE model.
    It explains exactly which input (Maturity, Strike, or Path Roughness) 
    is driving the current Option Price and Delta.
    """
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DeepBSDE_RoughVol().to(self.device)
        
        if model_path is None:
            model_path = os.path.join(BASE_DIR, "Data", "DeepBSDE_empirical.pth")
            
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def attribute(self, historical_paths, contract_terms):
        """
        Computes gradients of the predicted price with respect to inputs.
        
        Input:
            historical_paths: (1, seq_len, 2)
            contract_terms: (1, 2) -> [Maturity, Strike]
            
        Returns:
            dict: Attribution scores for Maturity, Strike, and Path Jagginess.
        """
        historical_paths = historical_paths.clone().detach().requires_grad_(True)
        contract_terms = contract_terms.clone().detach().requires_grad_(True)
        
        # 1. Forward Pass to get Price
        price, _ = self.model(historical_paths, contract_terms)
        
        # 2. Backward Pass for Price Saliency
        price.backward(torch.ones_like(price))
        
        # Extract gradients
        grad_contract = contract_terms.grad[0].cpu().numpy()
        grad_paths = historical_paths.grad[0].cpu().numpy()
        
        # Sensitivity to T and K
        attr_maturity = abs(grad_contract[0])
        attr_strike = abs(grad_contract[1])
        
        # Sensitivity to Roughness (sum of absolute gradients across the 20-day window)
        # We look at the 'Spot' channel gradients (channel 0)
        attr_roughness = np.sum(np.abs(grad_paths[:, 0])) 
        
        # Normalize for visualization
        total = attr_maturity + attr_strike + attr_roughness + 1e-9
        
        scores = {
            'Maturity (T)': attr_maturity / total,
            'Strike (K)': attr_strike / total,
            'Path Roughness (H)': attr_roughness / total
        }
        
        return scores

def analyze_decision_drivers():
    """
    Runs the explainer on the current empirical market state.
    """
    print("\n[QUANT ARMOURY] Executing Model Interpretability Analysis...")
    
    try:
        X_p, X_c, _, _, _, _ = prepare_empirical_batches(seq_len=20)
    except:
        print("Error: No empirical data found. Run pipeline first.")
        return
        
    explainer = DeepExplainer()
    
    # Analyze a few samples (e.g. at-the-money vs out-of-the-money)
    sample_indices = [0, len(X_c)//2] # First and Middle contract
    
    results = []
    for idx in sample_indices:
        p_tnsr = X_p[idx].unsqueeze(0).to(explainer.device)
        c_tnsr = X_c[idx].unsqueeze(0).to(explainer.device)
        
        attribution = explainer.attribute(p_tnsr, c_tnsr)
        results.append({
            'contract_idx': idx,
            'maturity': X_c[idx, 0].item(),
            'attribution': attribution
        })
        
    # Save attribution data for charting
    np.save(os.path.join(BASE_DIR, "Data", "model_attribution.npy"), results)
    print("[SUCCESS] Feature attribution analysis saved to Data/model_attribution.npy")

if __name__ == "__main__":
    analyze_decision_drivers()
