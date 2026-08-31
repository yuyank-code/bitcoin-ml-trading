"""Secure Binance Spot Testnet adapter.

Credentials are read only from environment variables. This module defaults to
DRY_RUN=true and refuses live endpoints unless BINANCE_TESTNET=true. It is
intended for integration testing of the model, not real-money execution.
"""
from __future__ import annotations
import hashlib
import hmac
import os
import time
from urllib.parse import urlencode
import requests

TESTNET_BASE = "https://testnet.binance.vision"
LIVE_BASE = "https://api.binance.com"

class BinanceConfig:
    def __init__(self):
        self.api_key=os.getenv("BINANCE_API_KEY","").strip()
        self.api_secret=os.getenv("BINANCE_API_SECRET","").strip()
        self.testnet=os.getenv("BINANCE_TESTNET","true").lower() in {"1","true","yes"}
        self.dry_run=os.getenv("BINANCE_DRY_RUN","true").lower() in {"1","true","yes"}
        if not self.testnet:
            raise RuntimeError("Live Binance is disabled by this adapter. Set BINANCE_TESTNET=true.")
        self.base=TESTNET_BASE

class BinanceTestnetClient:
    def __init__(self, config: BinanceConfig|None=None, timeout=10):
        self.cfg=config or BinanceConfig(); self.timeout=timeout
        self.session=requests.Session()
        if self.cfg.api_key: self.session.headers.update({"X-MBX-APIKEY":self.cfg.api_key})

    def _public(self,path,params=None):
        r=self.session.get(self.cfg.base+path,params=params or {},timeout=self.timeout); r.raise_for_status(); return r.json()

    def _signed(self,path,params=None,method="GET"):
        if not self.cfg.api_key or not self.cfg.api_secret:
            raise RuntimeError("BINANCE_API_KEY and BINANCE_API_SECRET are required for authenticated Testnet calls")
        p=dict(params or {}); p["timestamp"]=int(time.time()*1000); p["recvWindow"]=5000
        query=urlencode(p,doseq=True)
        p["signature"]=hmac.new(self.cfg.api_secret.encode(),query.encode(),hashlib.sha256).hexdigest()
        url=self.cfg.base+path
        r=self.session.request(method,url,params=p,timeout=self.timeout); r.raise_for_status(); return r.json()

    def ping(self): return self._public("/api/v3/ping")
    def server_time(self): return self._public("/api/v3/time")
    def account(self): return self._signed("/api/v3/account")
    def open_orders(self,symbol="BTCUSDT"): return self._signed("/api/v3/openOrders",{"symbol":symbol})
    def test_order(self,symbol,side,order_type="MARKET",quantity=None,quote_order_qty=None):
        """Validate an order on Testnet without creating a real order."""
        p={"symbol":symbol,"side":side,"type":order_type}
        if quantity is not None: p["quantity"]=quantity
        if quote_order_qty is not None: p["quoteOrderQty"]=quote_order_qty
        return self._signed("/api/v3/order/test",p,"POST")

    def create_order(self,**params):
        """Create a Testnet order only when DRY_RUN=false; never reaches live Binance."""
        if self.cfg.dry_run:
            return {"dry_run":True,"endpoint":"/api/v3/order","params":params}
        return self._signed("/api/v3/order",params,"POST")

if __name__=="__main__":
    c=BinanceTestnetClient()
    print(c.ping())
    print(c.server_time())
    print("DRY_RUN:",c.cfg.dry_run)
