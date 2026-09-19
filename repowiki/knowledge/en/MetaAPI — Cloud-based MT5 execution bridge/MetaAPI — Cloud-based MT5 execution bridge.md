---
kind: external_dependency
name: MetaAPI — Cloud-based MT5 execution bridge
slug: metaapi
category: external_dependency
category_hints:
    - vendor_identity
    - auth_protocol
scope:
    - '**'
---


Auth shape: token + account ID passed at runtime; no secrets committed to source. Region is auto-detected from the linked MT5 account.