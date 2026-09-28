powershell -Command "$p='e:\Desktop\NeoMind\Bazz\autonoumuse_trader\features\timeframe_features.py'; $c=Get-Content -Raw -LiteralPath $p; $old='    # 14. Volume ratio (current / 20-period average)
    avg_volume = df[''volume''].rolling(20).mean()
    result[f''{tf_name}_volume_ratio''] = df[''volume''] / avg_volume'; $new='    # 14. Volume ratio (current / 20-period average)
    # MT5 supplies tick_volume, CSVs supply volume — handle BOTH (P1 fix).
    vol_col = ''volume'' if ''volume'' in df.columns else (''tick_volume'' if ''tick_volume'' in df.columns else None)
    avg_volume = df[vol_col].rolling(20).mean() if vol_col else None
    if vol_col:
        result[f''{tf_name}_volume_ratio''] = df[vol_col] / avg_volume
    else:
        result[f''{tf_name}_volume_ratio''] = 1.0'; if($c.Contains($old)){ $c=$c.Replace($old,$new); [IO.File]::WriteAllText($p,$c,[Text.UTF8Encoding]::new($false)); Write-Output 'OK timeframe_features volume fix' } else { Write-Output 'MISS' }"