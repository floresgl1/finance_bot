"""Test configuration — mock external dependencies unavailable in CI."""

import sys
from unittest.mock import MagicMock

# alpaca-trade-api has a transitive dependency (msgpack) that fails to build
# from source in some CI environments. Since every test mocks the API client
# anyway, we inject a stub module before any test file imports live_trader.
if "alpaca_trade_api" not in sys.modules:
    sys.modules["alpaca_trade_api"] = MagicMock()
