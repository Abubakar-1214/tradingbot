//+------------------------------------------------------------------+
//|                               TradingAI_TriConfluence_Scalper.mq5|
//|                        Autonomous AI Trader - MT5 Strategy Tester|
//+------------------------------------------------------------------+
#property copyright "Autonomous AI Trader"
#property link      "https://github.com"
#property version   "1.20"
#property description "Pure SMA (M15+M5+M1) Tri-Confluence Scalper with Dynamic Trend Runner ($10-$20 Target)"

#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>

//--- Input Parameters
input group "=== SMA Settings ===";
input int      InpFastSMA           = 20;       // Fast SMA Period
input int      InpSlowSMA           = 50;       // Slow SMA Period
input bool     InpCheckM5           = true;     // Confirm with M5 Trend
input bool     InpCheckM1           = true;     // Confirm with M1 Trend

input group "=== Scalp & Profit Settings ===";
input double   InpTargetProfitUSD   = 1.00;     // Base Target Profit in USD
input int      InpReenterBarsWait   = 2;        // M15 Bars to Skip After Close
input double   InpLotSize           = 0.01;     // Trade Volume
input ulong    InpMagic             = 22388333445; // EA Magic Number

input group "=== Dynamic Trend Runner ($10-$20 Engine) ===";
input bool     InpTrendRunnerEnabled = true;     // Run strong trends up to $10-$20
input double   InpRunnerMinProfitUSD = 1.00;     // Min Profit to evaluate continuation ($)
input double   InpRunnerMaxProfitUSD = 15.00;    // Max Big Target Profit ($10 - $20)
input double   InpRunnerTrailPct     = 25.0;
input double   InpRunnerTrailStartUSD = 3.00;    // Min Profit before activating Trailing Stop ($)     // Trailing Pullback Lock from Peak (%)

input group "=== Multi-Trade & Balance Guard ===";
input int      InpMaxOpenPositions = 4;        // Max Simultaneous Positions (Hedge/Multi)
input double   InpMultiTradeMinEquity = 80.0;   // Min Equity Required for Additional Trade ($)

//--- Global Variables
CTrade         trade;
CPositionInfo  posInfo;

int            hFastM15, hSlowM15;
int            hFastM5,  hSlowM5;
int            hFastM1,  hSlowM1;

datetime       lastCloseBarTime = 0;
datetime       lastCloseRealTime = 0;
int            barsSinceClose = 999;
datetime       lastM15BarTime = 0;

// Peak profit tracker per ticket
ulong          gTickets[10];
double         gPeakProfits[10];

double GetPeakProfit(ulong ticket, double currentProfit)
{
   for(int i=0; i<10; i++)
   {
      if(gTickets[i] == ticket)
      {
         if(currentProfit > gPeakProfits[i]) gPeakProfits[i] = currentProfit;
         return gPeakProfits[i];
      }
   }
   for(int i=0; i<10; i++)
   {
      if(gTickets[i] == 0)
      {
         gTickets[i] = ticket;
         gPeakProfits[i] = currentProfit;
         return currentProfit;
      }
   }
   return currentProfit;
}

void RemoveTicket(ulong ticket)
{
   for(int i=0; i<10; i++)
   {
      if(gTickets[i] == ticket)
      {
         gTickets[i] = 0;
         gPeakProfits[i] = 0.0;
         break;
      }
   }
}

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   trade.SetExpertMagicNumber(InpMagic);
   trade.SetMarginMode();
   trade.SetTypeFillingBySymbol(_Symbol);

   for(int i=0; i<10; i++) { gTickets[i] = 0; gPeakProfits[i] = 0.0; }

   // Indicators on M15
   hFastM15 = iMA(_Symbol, PERIOD_M15, InpFastSMA, 0, MODE_SMA, PRICE_CLOSE);
   hSlowM15 = iMA(_Symbol, PERIOD_M15, InpSlowSMA, 0, MODE_SMA, PRICE_CLOSE);

   // Indicators on M5
   hFastM5  = iMA(_Symbol, PERIOD_M5,  InpFastSMA, 0, MODE_SMA, PRICE_CLOSE);
   hSlowM5  = iMA(_Symbol, PERIOD_M5,  InpSlowSMA, 0, MODE_SMA, PRICE_CLOSE);

   // Indicators on M1
   hFastM1  = iMA(_Symbol, PERIOD_M1,  InpFastSMA, 0, MODE_SMA, PRICE_CLOSE);
   hSlowM1  = iMA(_Symbol, PERIOD_M1,  InpSlowSMA, 0, MODE_SMA, PRICE_CLOSE);

   if(hFastM15 == INVALID_HANDLE || hSlowM15 == INVALID_HANDLE)
   {
      Print("Error creating M15 MA indicators!");
      return(INIT_FAILED);
   }

   Print("Autonomous AI Scalper (v1.20 Dynamic Trend Runner) initialized successfully on ", _Symbol);
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   IndicatorRelease(hFastM15);
   IndicatorRelease(hSlowM15);
   IndicatorRelease(hFastM5);
   IndicatorRelease(hSlowM5);
   IndicatorRelease(hFastM1);
   IndicatorRelease(hSlowM1);
}

//+------------------------------------------------------------------+
//| Helper: Get SMA Trend (1=BUY, 2=SELL, 0=None)                    |
//+------------------------------------------------------------------+
int GetTrend(int hFast, int hSlow)
{
   double fast[], slow[];
   ArraySetAsSeries(fast, true);
   ArraySetAsSeries(slow, true);

   if(CopyBuffer(hFast, 0, 1, 2, fast) < 2) return 0;
   if(CopyBuffer(hSlow, 0, 1, 2, slow) < 2) return 0;

   if(fast[0] > slow[0]) return 1; // BUY
   if(fast[0] < slow[0]) return 2; // SELL
   return 0;
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   // Track M15 Bar Transitions
   datetime currentM15Bar = iTime(_Symbol, PERIOD_M15, 0);
   if(currentM15Bar != lastM15BarTime)
   {
      lastM15BarTime = currentM15Bar;
      barsSinceClose++;
   }

   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   int myPositionsCount = 0;
   bool hasBuy = false;
   bool hasSell = false;

   int trendM15 = GetTrend(hFastM15, hSlowM15);
   int trendM5  = InpCheckM5 ? GetTrend(hFastM5, hSlowM5) : trendM15;
   int trendM1  = InpCheckM1 ? GetTrend(hFastM1, hSlowM1) : trendM15;

   // 1. Manage Each Open Position INDEPENDENTLY (Dynamic Trend Runner)
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(posInfo.SelectByIndex(i))
      {
         if(posInfo.Symbol() == _Symbol && posInfo.Magic() == InpMagic)
         {
            myPositionsCount++;
            int posType = (posInfo.PositionType() == POSITION_TYPE_BUY) ? 1 : 2;
            if(posType == 1) hasBuy = true;
            if(posType == 2) hasSell = true;

            ulong ticket = posInfo.Ticket();
            double profit = posInfo.Profit();
            double peakProfit = GetPeakProfit(ticket, profit);

            // A. Big Target Reached ($10 - $20)?
            if(InpTrendRunnerEnabled && profit >= InpRunnerMaxProfitUSD)
            {
               PrintFormat("🏆 [MAX TREND PROFIT] Ticket %I64u reached +$%.2f >= $%.2f! Closing to bank massive profit!",
                           ticket, profit, InpRunnerMaxProfitUSD);
               trade.PositionClose(ticket);
                  PlaySound("ok.wav");
               RemoveTicket(ticket);
               barsSinceClose = 0;
               lastCloseBarTime = currentM15Bar;
               return;
            }

            // B. Reached Minimum Profit ($1.00)?
            if(profit >= InpRunnerMinProfitUSD)
            {
               if(!InpTrendRunnerEnabled)
               {
                  PrintFormat("💰 [PROFIT LOCK] Ticket %I64u profit=+$%.2f. Closing immediately!", ticket, profit);
                  trade.PositionClose(ticket);
                  PlaySound("ok.wav");
                  RemoveTicket(ticket);
                  barsSinceClose = 0;
                  lastCloseBarTime = currentM15Bar;
                  return;
               }

               // Check if next candle is likely to go against trade (M1/M5 conflict)
               bool reversalRisk = false;
               if(trendM1 != 0 && trendM1 != posType) reversalRisk = true;
               if(trendM5 != 0 && trendM5 != posType) reversalRisk = true;

               // Check trailing pullback
               bool isPullingBack = false;
               if(peakProfit >= InpRunnerTrailStartUSD)
               {
                  double allowedDrop = MathMax(1.0, peakProfit * (InpRunnerTrailPct / 100.0));
                  if(profit <= (peakProfit - allowedDrop)) isPullingBack = true;
               }
               else if(peakProfit >= 2.00 && profit <= 0.75)
               {
                  isPullingBack = true;
               }

               if(reversalRisk)
               {
                  PrintFormat("⚡ [QUICK PROFIT EXIT] Ticket %I64u profit=+$%.2f. Next candle risk detected on M1/M5! Taking profit now!",
                              ticket, profit);
                  trade.PositionClose(ticket);
                  PlaySound("ok.wav");
                  RemoveTicket(ticket);
                  barsSinceClose = 0;
                  lastCloseBarTime = currentM15Bar;
                  return;
               }
               else if(isPullingBack)
               {
                  PrintFormat("🎯 [TRAILING PROFIT SECURED] Ticket %I64u profit=+$%.2f (peak +$%.2f) pulling back! Securing runner profit!",
                              ticket, profit, peakProfit);
                  trade.PositionClose(ticket);
                  PlaySound("ok.wav");
                  RemoveTicket(ticket);
                  barsSinceClose = 0;
                  lastCloseBarTime = currentM15Bar;
                  return;
               }
               // Else: Trend is strong & aligned! Let it run!
            }
         }
      }
   }

   // 2. Multi-Trade / Hedge Entry Check
   int allowedPositions = (equity > InpMultiTradeMinEquity) ? InpMaxOpenPositions : 1;
   if(myPositionsCount >= allowedPositions)
      return;

   // Rule A: Do not re-enter on exact same M15 candle where trade closed
   if(lastCloseBarTime != 0 && currentM15Bar == lastCloseBarTime)
      return;

   // Rule B: Wait at least InpReenterBarsWait M15 bars (if no positions held)
   if(myPositionsCount == 0 && barsSinceClose < InpReenterBarsWait)
      return;

   // Hedge Detection: If an existing position is in drawdown and M5/M1 confirm opposite trend:
   bool triggerHedgeSell = (hasBuy && myPositionsCount < allowedPositions && (trendM5 == 2 || trendM1 == 2));
   bool triggerHedgeBuy  = (hasSell && myPositionsCount < allowedPositions && (trendM5 == 1 || trendM1 == 1));

   if(!triggerHedgeSell && !triggerHedgeBuy && trendM15 == 0) return;

   // Open new trade with trend direction or hedge
   if((trendM15 == 1 && trendM5 == 1 && trendM1 == 1) || triggerHedgeBuy)
   {
      if(hasBuy && !triggerHedgeBuy) return;
      double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      trade.Buy(InpLotSize, _Symbol, ask, 0, 0, "AI_M15_TriConfluence");
      PrintFormat("⚡ [BUY ENTRY] Opened BUY at %.5f (Positions=%d | Equity=$%.2f)", ask, myPositionsCount + 1, equity);
   }
   else if(trendM15 == 2 && trendM5 == 2 && trendM1 == 2)
   {
      if(hasSell) return;
      double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      trade.Sell(InpLotSize, _Symbol, bid, 0, 0, "AI_M15_TriConfluence");
      PrintFormat("⚡ [SELL ENTRY] Opened SELL at %.5f (Positions=%d | Equity=$%.2f)", bid, myPositionsCount + 1, equity);
   }
}
//+------------------------------------------------------------------+
