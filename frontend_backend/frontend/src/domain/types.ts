export type Availability = 'available' | 'unavailable' | 'not_checked' | 'error'

export interface ApiEnvelope<T> {
  availability: Availability
  source?: string
  updated_at?: string
  data?: T
  reason?: string
}

export interface RuntimeGate {
  name: string
  availability: Availability
  passed?: boolean
  reason?: string
}

export interface DashboardStatus {
  trading_mode: 'demo' | 'live' | 'unavailable'
  trading_mode_source: string
  signal_source: 'rule' | 'model' | 'unavailable'
  require_promoted_model: boolean | null
  production_manifest: Availability
  production_manifest_path: string
  evaluations_passed: number | null
  evaluations_inspected: number | null
  supervisor: Availability
  kill_switch_armed: boolean | null
  runtime_gates: RuntimeGate[]
}

export interface EvaluationArtifact {
  run_name: string
  model_type: string
  source_path: string
  evaluated_at: string
  test_start?: string
  test_end?: string
  contract_hash?: string
  metrics: Record<string, number>
  thresholds: Record<string, number>
  passed: boolean
  failures: string[]
}

export interface ResumableModel {
  run_name: string
  model_type: string
  latest_checkpoint: string
  latest_checkpoint_name: string
  latest_step: number
  window: number
  n_features?: number
  has_macro?: boolean
  train_end: string
  allow_short: boolean
  has_model_zip: boolean
  has_evaluation: boolean
  checkpoints_count: number
}

export interface OhlcvBar {
  time: number
  open: number
  high: number
  low: number
  close: number
  volume?: number
}

export interface MarketBars {
  symbol: string
  timeframe: string
  bars: OhlcvBar[]
  last_bar_time: string
  historical: true
}

export interface RiskSnapshot {
  balance?: number
  mid?: number
  current_equity?: number
  peak_equity?: number
  daily_pnl?: number
  daily_loss_ratio?: number
  trades_today?: number
  consecutive_losses?: number
  halt_remaining_sec?: number
  trade_date?: string | null
  rejection_reasons?: Record<string, number>
  statistics_availability: Availability
  statistics_reason?: string
  config: Record<string, string | number | boolean>
  config_source: string
}

export interface Position {
  ticket: number | string
  side: string
  volume: number
  entry: number
  sl?: number | null
  tp?: number | null
  current_price?: number | null
  pnl?: number | null
}

export interface ClosedTrade {
  ticket: number | string
  side: string
  volume: number
  entry: number
  exit: number
  pnl: number
  reason: string
}

export interface Reconciliation {
  availability: Availability
  reason?: string
  ok?: boolean
  matched?: number
  mismatched?: number
  drift?: number
  timestamp?: string
}

export interface JobSnapshot {
  job_id: string
  status: 'starting' | 'running' | 'stopping' | 'unverified' | 'interrupted' | 'finished' | 'failed' | 'killed'
  pid?: number
  step?: number
  total_steps?: number
  started_at?: string
  return_code?: number | null
  log_path?: string
  preset?: string
  recovery_state?: string
}

export interface PreflightItem {
  id: string
  label: string
  availability: Availability
  passed?: boolean
  reason?: string
}

export const RISK_FIELDS = [
  { key: 'MAX_DAILY_LOSS', label: 'Maximum daily loss', kind: 'fraction' },
  { key: 'MAX_DRAWDOWN', label: 'Maximum drawdown', kind: 'fraction' },
  { key: 'MAX_CONSECUTIVE_LOSSES', label: 'Maximum consecutive losses', kind: 'integer' },
  { key: 'MAX_RISK_PER_TRADE', label: 'Maximum risk per trade', kind: 'fraction' },
  { key: 'MAX_POSITION', label: 'Maximum position', kind: 'fraction' },
  { key: 'MAX_TRADES_PER_DAY', label: 'Maximum trades per day', kind: 'integer' },
  { key: 'MIN_TRADE_INTERVAL_SEC', label: 'Minimum trade interval', kind: 'number' },
  { key: 'VOL_THRESHOLD', label: 'Volatility threshold', kind: 'number' },
  { key: 'MAX_SPREAD', label: 'Maximum spread', kind: 'number' },
  { key: 'MARKET_HOURS_ONLY', label: 'Market hours only', kind: 'boolean' },
  { key: 'CORRELATION_GUARD_ENABLED', label: 'Correlation guard', kind: 'boolean' },
  { key: 'CORRELATION_ASSET', label: 'Correlation asset', kind: 'text' },
  { key: 'CORRELATION_BLOCK_LONG_ON_UP', label: 'Long block threshold', kind: 'number' },
  { key: 'MIN_CONFIDENCE', label: 'Minimum confidence', kind: 'fraction' },
  { key: 'MIN_ENSEMBLE_AGREEMENT', label: 'Minimum ensemble agreement', kind: 'fraction' },
  { key: 'USE_KELLY', label: 'Use Kelly sizing', kind: 'boolean' },
  { key: 'KELLY_FRACTION', label: 'Kelly fraction', kind: 'fraction' },
  { key: 'KELLY_MIN_TRADES', label: 'Kelly minimum trades', kind: 'integer' },
] as const

export type RiskFieldKey = (typeof RISK_FIELDS)[number]['key']
export const RISK_REASONS = [
  'APPROVED', 'CIRCUIT_BREAKER', 'HALTED', 'MAX_DRAWDOWN', 'POSITION_TOO_LARGE',
  'TOO_MANY_LOSSES', 'HIGH_VOLATILITY', 'CORRELATION_GUARD', 'EVENT_RISK',
  'MAX_TRADES', 'COOLDOWN', 'SPREAD_TOO_WIDE', 'MARKET_CLOSED',
] as const

export const PRESETS = [
  ['dreamer', 'train/train_dreamer.py'],
  ['transformer', 'train/train_transformer.py'],
  ['ppo', 'train/train_ppo.py'],
  ['ensemble', 'train/train_ensemble.py'],
  ['god-mode', 'train/train_god_mode.py'],
  ['ultimate-150', 'train/train_ultimate_150.py'],
  ['adversarial', 'train/train_adversarial.py'],
  ['meta-train', 'train/meta_train_dreamer.py'],
] as const
