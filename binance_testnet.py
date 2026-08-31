"""Secure Binance Spot Testnet adapter.

Credentials are loaded from the local environment/.env file. The adapter is
hard-locked to Binance Spot Testnet and defaults to dry-run. It never uses the
live Binance endpoint.
"""
from __future__ import annotations
import hashlib
import hmac
import os
import time
from urllib.parse import urlencode
import requests
from dotenv import load_dotenv

load_dotenv()

TESTNET_BASE = "https://testnet.binance.vision"

class BinanceConfig:
    def __init__(self):
        self.api_key=os.getenv("BINANCE_API_KEY","").strip()
        self.api_secret=os.getenv("BINANCE_API_SECRET","").strip()
        self.testnet=os.getenv("BINANCE_TESTNET","true").lower() in {"1","true","yes"}
        self.dry_run=os.getenv("BINANCE_DRY_RUN","true").lower() in {"1","true","yes"}
        if not self.testnet:
            raise RuntimeError("Safety lock: BINANCE_TESTNET must remain true.")
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
            raise RuntimeError("BINANCE_API_KEY and BINANCE_API_SECRET are required")
        p=dict(params or {}); p["timestamp"]=int(time.time()*1000); p["recvWindow"]=5000
        query=urlencode(p,doseq=True)
        p["signature"]=hmac.new(self.cfg.api_secret.encode(),query.encode(),hashlib.sha256).hexdigest()
        r=self.session.request(method,self.cfg.base+path,params=p,timeout=self.timeout); r.raise_for_status(); return r.json()

    def ping(self): return self._public("/api/v3/ping")
    def server_time(self): return self._public("/api/v3/time")
    def account(self): return self._signed("/api/v3/account")
    def open_orders(self,symbol="BTCUSDT"): return self._signed("/api/v3/openOrders",{"symbol":symbol})
    def test_order(self,symbol,side,order_type="MARKET",quantity=None,quote_order_qty=None):
        p={"symbol":symbol,"side":side,"type":order_type}
        if quantity is not None: p["quantity"]=quantity
        if quote_order_qty is not None: p["quoteOrderQty"]=quote_order_qty
        return self._signed("/api/v3/order/test",p,"POST")

    def create_order(self,**params):
        if self.cfg.dry_run:
            return {"dry_run":True,"endpoint":"/api/v3/order","params":params}
        return self._signed("/api/v3/order",params,"POST")

if __name__=="__main__":
    c=BinanceTestnetClient()
    print("PING:",c.ping())
    print("SERVER_TIME:",c.server_time())
    print("TESTNET:",c.cfg.testnet)
    print("DRY_RUN:",c.cfg.dry_run)
    if c.cfg.api_key and c.cfg.api_secret:
        account=c.account()
        print("AUTHENTICATED: True")
        print("CAN_TRADE:",account.get("canTrade"))
        print("ACCOUNT_TYPE:",account.get("accountType"))
    else:
        print("AUTHENTICATED: False (credentials not loaded)")
