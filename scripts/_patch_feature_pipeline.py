"""Apply two REAL production fixes to core/feature_pipeline.py:

1. transform_feature_pipeline double-adds scaler_eps: fit stores std = std + eps
   in the contract, then transform divides by (std + eps) AGAIN — transform is
   numerically inconsistent with the training transform.  Fix: divide by the
   stored std as-is (it already contains eps; eps > 0 always, so no div-by-zero).

2. transform_feature_pipeline does not validate the contract SHAPE before use:
   a malformed contract (missing keys / wrong version) raises raw KeyError
   instead of the documented FeatureContractError.  Fix: shared
   _validate_contract_shape() used by load_feature_contract AND transform.
"""
from __future__ import annotations

from pathlib import Path

TARGET = Path(__file__).resolve().parent.parent / "core" / "feature_pipeline.py"

# 1. Insert the shape-validator helper before load_feature_contract.
HELPER = '''def _validate_contract_shape(contract: dict) -> None:
    """Raise FeatureContractError when a contract dict is malformed/unsupported."""
    required = {"version", "feature_names", "scaler", "pipeline_params"}
    missing = required - set(contract.keys())
    if missing:
        raise FeatureContractError(
            f"feature contract is malformed; missing keys: {sorted(missing)}"
        )
    if contract.get("version") != CONTRACT_VERSION:
        raise FeatureContractError(
            f"feature contract version {contract.get('version')} != expected {CONTRACT_VERSION}"
        )


def load_feature_contract(path: Union[str, Path]) -> dict:'''

OLD_LOAD = '''def load_feature_contract(path: Union[str, Path]) -> dict:'''

# 2. Replace the inline validation inside load_feature_contract with the helper.
OLD_VALIDATE = '''    required = {"version", "feature_names", "scaler", "pipeline_params"}
    missing = required - set(contract.keys())
    if missing:
        raise FeatureContractError(f"feature contract at {p} is malformed; missing keys: {sorted(missing)}")
    if contract.get("version") != CONTRACT_VERSION:
        raise FeatureContractError(
            f"feature contract version {contract.get('version')} != expected {CONTRACT_VERSION}"
        )
    return contract'''

NEW_VALIDATE = '''    _validate_contract_shape(contract)
    return contract'''

# 3. Call the shape validator in transform_feature_pipeline right after the
#    contract-is-None guard.
OLD_TRANSFORM_GUARD = '''    if contract is None:
        raise FeatureContractError(
            "transform_feature_pipeline requires a feature contract "
            "(fit_feature_pipeline first, then save/load it)."
        )
    cfg = cfg or FeatureConfig()'''

NEW_TRANSFORM_GUARD = '''    if contract is None:
        raise FeatureContractError(
            "transform_feature_pipeline requires a feature contract "
            "(fit_feature_pipeline first, then save/load it)."
        )
    _validate_contract_shape(contract)
    cfg = cfg or FeatureConfig()'''

# 4. Remove the double scaler_eps add in transform.
OLD_DIV = '''    X = ((X - mean) / (std + cfg.scaler_eps)).astype(np.float32)'''
NEW_DIV = '''    X = ((X - mean) / std).astype(np.float32)'''


def main() -> int:
    src = TARGET.read_text(encoding="utf-8")
    checks = 0

    if OLD_LOAD in src and HELPER not in src:
        src = src.replace(OLD_LOAD, HELPER, 1)
        checks += 1
        print("PATCHED [1] helper inserted before load_feature_contract")
    else:
        print("SKIP  [1] helper already present or anchor missing")

    if OLD_VALIDATE in src:
        src = src.replace(OLD_VALIDATE, NEW_VALIDATE, 1)
        checks += 1
        print("PATCHED [2] load_feature_contract now uses _validate_contract_shape")
    else:
        print("SKIP  [2] old inline validation not found (already patched?)")

    if OLD_TRANSFORM_GUARD in src:
        src = src.replace(OLD_TRANSFORM_GUARD, NEW_TRANSFORM_GUARD, 1)
        checks += 1
        print("PATCHED [3] transform_feature_pipeline validates contract shape")
    else:
        print("SKIP  [3] transform guard anchor not found")

    if OLD_DIV in src:
        src = src.replace(OLD_DIV, NEW_DIV, 1)
        checks += 1
        print("PATCHED [4] transform no longer double-adds scaler_eps")
    else:
        print("SKIP  [4] double-eps division not found")

    if checks == 4:
        TARGET.write_text(src, encoding="utf-8")
        print(f"WROTE {TARGET} ({len(src)} chars)")
        return 0
    print(f"ERROR: applied only {checks}/4 replacements — refusing to write partial file")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
