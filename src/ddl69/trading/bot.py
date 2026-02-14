"""Automated paper trading bot with P&L tracking"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional
from enum import Enum
import json

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class Order:
    """Single trade execution"""

    def __init__(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        price: float,
        signal_confidence: float,
    ):
        self.order_id = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        self.symbol = symbol
        self.side = side
        self.quantity = quantity
        self.price = price
        self.signal_confidence = signal_confidence
        self.entry_time = datetime.now(timezone.utc)
        self.exit_price = None
        self.exit_time = None
        self.status = "OPEN"  # OPEN, CLOSED, CANCELLED

    def close(self, exit_price: float) -> dict:
        """Close the position"""
        self.exit_price = exit_price
        self.exit_time = datetime.now(timezone.utc)
        self.status = "CLOSED"

        # Calculate P&L
        if self.side == OrderSide.BUY:
            pnl = (exit_price - self.price) * self.quantity
        else:  # SELL
            pnl = (self.price - exit_price) * self.quantity

        pnl_pct = pnl / (self.price * self.quantity)

        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "entry_price": self.price,
            "exit_price": exit_price,
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 4),
            "duration": (self.exit_time - self.entry_time).total_seconds(),
        }


class PortfolioPosition:
    """Position in a symbol"""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.quantity = 0
        self.avg_entry_price = 0
        self.orders = []

    def buy(self, quantity: int, price: float, confidence: float) -> Order:
        """Reduce or reverse short position, or go long"""
        order = Order(self.symbol, OrderSide.BUY, quantity, price, confidence)
        self.orders.append(order)

        if self.quantity >= 0:
            # Going longer
            total_cost = (self.avg_entry_price * self.quantity) + (price * quantity)
            self.quantity += quantity
            self.avg_entry_price = total_cost / self.quantity if self.quantity > 0 else 0
        else:
            # Covering short
            self.quantity += quantity
            if self.quantity > 0:
                self.avg_entry_price = price

        return order

    def sell(self, quantity: int, price: float, confidence: float) -> Order:
        """Close or reduce long position, or go short"""
        order = Order(self.symbol, OrderSide.SELL, quantity, price, confidence)
        self.orders.append(order)

        if self.quantity <= 0:
            # Going shorter
            self.quantity -= quantity
            self.avg_entry_price = price
        else:
            # Closing long
            self.quantity -= quantity
            if self.quantity < 0:
                self.avg_entry_price = price

        return order

    def get_unrealized_pnl(self, current_price: float) -> dict:
        """Calculate unrealized P&L"""
        if self.quantity == 0:
            return {"quantity": 0, "unrealized_pnl": 0, "unrealized_pct": 0}

        unrealized = (current_price - self.avg_entry_price) * self.quantity
        unrealized_pct = unrealized / (self.avg_entry_price * self.quantity) if self.avg_entry_price > 0 else 0

        return {
            "quantity": self.quantity,
            "avg_entry_price": round(self.avg_entry_price, 2),
            "current_price": round(current_price, 2),
            "unrealized_pnl": round(unrealized, 2),
            "unrealized_pct": round(unrealized_pct, 4),
        }


class PaperTradingBot:
    """Automated paper trading with position management"""

    def __init__(
        self,
        initial_capital: float = 100000,
        max_position_size: float = 0.1,  # 10% of portfolio per symbol
        risk_per_trade: float = 0.02,  # 2% risk per trade
    ):
        """
        Args:
          initial_capital: Starting cash
          max_position_size: Max percent of portfolio per position
          risk_per_trade: Max percent of capital to risk per trade
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.max_position_size = max_position_size
        self.risk_per_trade = risk_per_trade

        self.positions = {}  # {symbol: PortfolioPosition}
        self.trade_history = []
        self.start_time = datetime.now(timezone.utc)

    def calculate_position_size(
        self,
        symbol: str,
        current_price: float,
        signal_confidence: float,
    ) -> int:
        """Calculate position size using Kelly criterion variant

        Args:
          symbol: Trade symbol
          current_price: Current price
          signal_confidence: Model confidence (0-1)

        Returns:
          Number of shares to trade
        """
        # Risk amount
        risk_amount = self.initial_capital * self.risk_per_trade * signal_confidence

        # Max position dollar amount
        max_pos_value = self.initial_capital * self.max_position_size

        # Position size in shares
        position_shares = int(risk_amount / current_price)
        max_shares = int(max_pos_value / current_price)

        return min(position_shares, max_shares)

    def execute_signal(
        self,
        symbol: str,
        signal: str,  # BUY, SELL, HOLD
        probability: float,
        confidence: float,
        current_price: float,
    ) -> Optional[Order]:
        """Execute trade based on signal

        Returns:
          Order object or None if not executed
        """
        if signal == "HOLD" or confidence < 0.5:
            return None

        if symbol not in self.positions:
            self.positions[symbol] = PortfolioPosition(symbol)

        position = self.positions[symbol]
        quantity = self.calculate_position_size(symbol, current_price, confidence)

        if quantity == 0:
            logger.warning(f"Position size too small for {symbol}")
            return None

        try:
            if signal == "BUY":
                if position.quantity >= 0:  # Not short or flat
                    if self.cash >= current_price * quantity:
                        order = position.buy(quantity, current_price, confidence)
                        self.cash -= current_price * quantity
                        logger.info(f"BUY order: {quantity} {symbol} @ ${current_price}")
                        return order

            elif signal == "SELL":
                if position.quantity > 0:  # Currently long
                    order = position.sell(position.quantity, current_price, confidence)
                    self.cash += current_price * position.quantity
                    logger.info(f"SELL order: close {symbol} position @ ${current_price}")
                    return order

        except Exception as e:
            logger.error(f"Failed to execute signal for {symbol}: {e}")

        return None

    def get_portfolio_value(self, current_prices: dict[str, float]) -> float:
        """Calculate total portfolio value

        Args:
          current_prices: {symbol: price}

        Returns:
          Total portfolio value
        """
        return self.cash + sum(
            pos.quantity * current_prices.get(pos.symbol, 0)
            for pos in self.positions.values()
            if pos.symbol in current_prices
        )

    def get_portfolio_status(self, current_prices: dict[str, float]) -> dict:
        """Get full portfolio status"""
        total_value = self.get_portfolio_value(current_prices)
        total_pnl = total_value - self.initial_capital
        total_pnl_pct = total_pnl / self.initial_capital if self.initial_capital > 0 else 0

        positions_status = {}
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                positions_status[symbol] = position.get_unrealized_pnl(current_prices[symbol])

        # Win rate
        closed_trades = [o for o in self._get_all_orders() if o.status == "CLOSED"]
        winning_trades = sum(1 for o in closed_trades if self._get_trade_pnl(o) > 0)
        win_rate = winning_trades / len(closed_trades) if closed_trades else 0

        # Avg trade duration
        durations = [
            (o.exit_time - o.entry_time).total_seconds()
            for o in closed_trades
            if o.exit_time
        ]
        avg_duration = np.mean(durations) if durations else 0

        return {
            "portfolio_value": round(total_value, 2),
            "total_pnl": round(total_pnl, 2),
            "total_pnl_pct": round(total_pnl_pct, 4),
            "cash": round(self.cash, 2),
            "positions": positions_status,
            "total_trades": len(closed_trades),
            "win_rate": round(win_rate, 4),
            "avg_trade_duration": round(avg_duration, 0),
        }

    def _get_all_orders(self) -> list[Order]:
        """Get all orders across positions"""
        orders = []
        for position in self.positions.values():
            orders.extend(position.orders)
        return orders

    @staticmethod
    def _get_trade_pnl(order: Order) -> float:
        """Calculate P&L for a trade"""
        if order.status != "CLOSED" or order.exit_price is None:
            return 0

        if order.side == OrderSide.BUY:
            return (order.exit_price - order.price) * order.quantity
        else:
            return (order.price - order.exit_price) * order.quantity

    def get_performance_metrics(self) -> dict:
        """Calculate performance metrics"""
        closed_trades = [o for o in self._get_all_orders() if o.status == "CLOSED"]

        if not closed_trades:
            return {"error": "No closed trades"}

        pnls = [self._get_trade_pnl(o) for o in closed_trades]
        returns = [pnl / (o.price * o.quantity) for o, pnl in zip(closed_trades, pnls)]

        sharpe = (
            np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
            if returns else 0
        )

        return {
            "total_trades": len(closed_trades),
            "win_rate": round(sum(1 for p in pnls if p > 0) / len(pnls), 4),
            "avg_win": round(np.mean([p for p in pnls if p > 0] or [0]), 2),
            "avg_loss": round(np.mean([p for p in pnls if p <= 0] or [0]), 2),
            "sharpe_ratio": round(sharpe, 4),
            "total_pnl": round(sum(pnls), 2),
            "profit_factor": round(
                sum(p for p in pnls if p > 0) / abs(sum(p for p in pnls if p < 0) or 1),
                2,
            ),
        }
