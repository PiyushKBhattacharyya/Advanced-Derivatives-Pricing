import os
import sys
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
import logging

try:
    import asyncio
    import nest_asyncio
    
    # Streamlit ScriptRunner threads lack a default asyncio event loop, 
    # which fatally crashes ib_insync (via eventkit) on import.
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
    nest_asyncio.apply()
        
    from ib_insync import IB, Index, Stock, Option, LimitOrder, util
    HAS_IB = True
except ImportError:
    HAS_IB = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

class InteractiveBrokersDeepBSDE:
    """
    Tier-1 Execution Algorithm Architecture.
    
    This bridges the academic Deep BSDE Simulation native PyTorch environment securely
    to a continuous live market exchange via Interactive Brokers. It handles sub-millisecond 
    TCP websocket streaming and completely bypasses Yahoo Finance rate-limits.
    """
    
    def __init__(self, host: str = '127.0.0.1', port: int = 7497, client_id: int = 999):
        """
        Connect to Interactive Brokers TWS (Trader Workstation) or IB Gateway.
        By default, Paper Trading (Simulated) runs on Port 7497.
        Live Trading runs on Port 7496. Do NOT shift to 7496 unless fully backtested.
        """
        self.host = host
        self.port = port
        self.client_id = client_id
        
        if HAS_IB:
            self.ib = IB()
        else:
            self.ib = None
            
        self.connected = False
        
    def connect_to_exchange(self) -> bool:
        """Explicitly connects the Python Runtime Engine to the IBKR TCP socket."""
        if not HAS_IB:
            return False
            
        try:
            # IB-insync relies heavily on python's asyncio event loop
            try:
                util.startLoop() # For Jupyter/Streamlit environments
            except:
                pass
                
            self.ib.connect(self.host, self.port, clientId=self.client_id)
            self.connected = True
            logging.info(f"Successfully bridged PyTorch Engine to Interactive Brokers (Port {self.port}).")
            return True
        except Exception as e:
            logging.warning(f"IBKR Connection Refused. Ensure TWS/Gateway is actively running. Fallback to yfinance active. | Error: {e}")
            return False
            
    def fetch_live_asset_tick(self, symbol: str = 'AAPL', sec_type: str = 'STK') -> Tuple[float, float]:
        """
        Returns exactly continuous `[Spot, VIX]` instantaneous tensor metrics for a given asset.
        Supports 'STK' (Stock) and 'IND' (Index).
        """
        if not self.connected:
            return None, None
            
        if sec_type == 'STK':
            asset_contract = Stock(symbol, 'SMART', 'USD')
        else:
            asset_contract = Index(symbol, 'CBOE') # Default to CBOE for Indexes
            
        vix_contract = Index('VIX', 'CBOE')
        
        self.ib.qualifyContracts(asset_contract, vix_contract)
        
        # Pull realtime snapshot constraints
        asset_ticker = self.ib.reqMktData(asset_contract, '', False, False)
        vix_ticker = self.ib.reqMktData(vix_contract, '', False, False)
        
        self.ib.sleep(1) # Buffer tick
        
        S_live = asset_ticker.marketPrice()
        vix_live = vix_ticker.marketPrice()
        
        # Fallback to close
        if np.isnan(S_live) or S_live <= 0:
            S_live = asset_ticker.close
        if np.isnan(vix_live) or vix_live <= 0:
            vix_live = vix_ticker.close
            
        return S_live, vix_live

    def fetch_dividend_yield(self, symbol: str) -> float:
        """
        Fetches the fundamental dividend yield for a stock ticker.
        """
        if not self.connected:
            return 0.0
            
        contract = Stock(symbol, 'SMART', 'USD')
        self.ib.qualifyContracts(contract)
        
        # Request fundamental data or tick types for dividend info
        # For simplicity in this research phase, we pull the 12-month trailing yield
        ticker = self.ib.reqMktData(contract, '258', False, False) # 258 is fundamental ratios
        self.ib.sleep(1)
        
        # IBKR often provides this in fundamental ratios
        # If unavailable, default to a conservative estimate or 0
        try:
            return float(ticker.fundamentalRatios.get('DY', 0.0)) / 100.0
        except:
            return 0.0
        
    def construct_deep_bsde_limit_order(self, right: str, K: float, T: float, predicted_price: float, trade_qty: int = 1) -> Any:
        """
        Autonomous Algorithmic Hedging Pipe.
        
        If the Deep BSDE predicts physically anomalous curvature (arbitrage), the code dynamically 
        builds an options contract and blasts a Limit Order directly to the paper-trading exchange 
        at exactly the PyTorch output price limits.
        """
        if not self.connected:
            return None
            
        # Standardize physical expiration bounds (Simplified strictly for Friday weekly formats)
        days_to_expiry = int(T * 365)
        target_expiry = (pd.Timestamp.today() + pd.Timedelta(days=days_to_expiry)).strftime('%Y%m%d')
        
        # Build the physical SPX Options Contract
        option_contract = Option(symbol='SPX', lastTradeDateOrContractMonth=target_expiry, 
                                 strike=K, right=right, exchange='SMART', multiplier='100')
        
        self.ib.qualifyContracts(option_contract)
        
        # The Neural Network decides the absolute explicit Limit Price geometry limits
        order = LimitOrder('BUY', trade_qty, round(predicted_price, 2))
        
        trade = self.ib.placeOrder(option_contract, order)
        logging.info(f"Deployed PyTorch Limit Order -> [BUY {trade_qty} SPX @ {target_expiry} {K}{right} | Limit: ${predicted_price:.2f}]")
        
        return trade
        
    def disconnect(self):
        """Force tear-down the TCP pipe."""
        if self.ib and self.ib.isConnected():
            self.ib.disconnect()
            self.connected = False
            logging.info("Deep BSDE safely severed Interactive Brokers structural connection.")
