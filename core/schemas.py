from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

class OptionContract(BaseModel):
    contract_symbol: str
    option_type: str = Field(pattern="^(call|put)$")
    strike: float
    last_price: float
    bid: float
    ask: float
    implied_volatility: float
    open_interest: int
    volume: int
    expiration_date: datetime
    days_to_expiry: float

class MarketData(BaseModel):
    ticker: str
    spot_price: float
    risk_free_rate: float = 0.05
    dividend_yield: float = 0.0