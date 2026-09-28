from dataclasses import dataclass

import numpy as np

ACCOUNT_STATE_DIM = 5


@dataclass(frozen=True)
class AccountState:
    position: int
    trade_pnl: float
    bars_in_trade: int
    drawdown: float
    equity_ratio: float

    def to_vector(self) -> np.ndarray:
        return np.array(
            [
                float(self.position),
                float(np.clip(self.trade_pnl * 100.0, -10.0, 10.0)),
                float(np.log1p(self.bars_in_trade)) / 5.0,
                float(self.drawdown),
                float(np.log(max(self.equity_ratio, 1e-6))),
            ],
            dtype=np.float32,
        )


def build_observation(features_window: np.ndarray, account: AccountState) -> np.ndarray:
    window = np.asarray(features_window, dtype=np.float32)
    if window.ndim != 2:
        raise ValueError("features_window must have shape (window, n_features)")
    return np.concatenate((window.reshape(-1), account.to_vector())).astype(np.float32)


def obs_dim(window: int, n_features: int) -> int:
    if window < 1 or n_features < 1:
        raise ValueError("window and n_features must be positive")
    return int(window) * int(n_features) + ACCOUNT_STATE_DIM
