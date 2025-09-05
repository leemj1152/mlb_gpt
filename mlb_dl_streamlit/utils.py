from __future__ import annotations
import mlbstatsapi
from functools import lru_cache

@lru_cache(maxsize=1)
def get_mlb_client() -> mlbstatsapi.Mlb:
    # 하나의 세션을 재사용
    return mlbstatsapi.Mlb()
