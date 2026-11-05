"""
MT5 Strategy Tester Launcher
Launches the native MT5 Strategy Tester with TradingAI_TriConfluence_Scalper.ex5
"""
import os
import subprocess
from datetime import datetime, timedelta

MT5_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"
INI_PATH = os.path.abspath("tester_config.ini")

def create_tester_ini(
    symbol: str = "BTCUSD",
    period: str = "M15",
    deposit: float = 100.0,
    leverage: int = 100,
    days: int = 30,
    visual: bool = True
):
    to_date = datetime.now()
    from_date = to_date - timedelta(days=days)

    ini_content = f"""[Tester]
Expert=TradingAI_TriConfluence_Scalper.ex5
Symbol={symbol}
Period={period}
Deposit={deposit:.2f}
Currency=USD
Leverage={leverage}
Model=1
ExecutionMode=0
Optimization=0
FromDate={from_date.strftime('%Y.%m.%d')}
ToDate={to_date.strftime('%Y.%m.%d')}
ForwardMode=0
Report=TesterReport_{symbol}_{period}
ReplaceReport=1
ShutdownTerminal=0
Visual={1 if visual else 0}

[TesterInputs]
InpFastSMA=20
InpSlowSMA=50
InpCheckM5=true
InpCheckM1=true
InpTargetProfitUSD=0.40
InpReenterBarsWait=2
InpLotSize=0.01
InpMagic=22388333445
"""
    with open(INI_PATH, "w", encoding="utf-8") as f:
        f.write(ini_content)
    print(f"Created MT5 Tester configuration: {INI_PATH}")
    return INI_PATH

def launch_tester():
    ini_file = create_tester_ini()
    cmd = f'"{MT5_PATH}" /config:"{ini_file}"'
    print(f"Launching MT5 Strategy Tester with command:\n{cmd}")
    subprocess.Popen(cmd, shell=True)
    print("\n✅ MT5 Strategy Tester started! Check your MetaTrader 5 window.")

if __name__ == "__main__":
    launch_tester()

