import os
from typing import Any, Dict, Optional
from urllib.parse import quote

import aiohttp


class GreenlandGameClient:
    """Internal HTTP adapter behind authenticated Atlantis game tools."""

    def __init__(self) -> None:
        self.base_url = os.getenv(
            "GREENLAND_GAME_URL", ""
        ).rstrip("/")
        self.authority_token = os.getenv("GAME_SERVER_AUTHORITY_TOKEN", "").strip()

    async def request(
        self,
        method: str,
        path: str,
        *,
        body: Optional[Dict[str, Any]] = None,
        authority: bool = True,
    ) -> Any:
        if not self.base_url:
            raise RuntimeError('GREENLAND_GAME_URL must explicitly select the world service')
        headers = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        if authority:
            if not self.authority_token:
                raise RuntimeError(
                    "GAME_SERVER_AUTHORITY_TOKEN is not configured on greenland-game"
                )
            headers["Authorization"] = f"Bearer {self.authority_token}"

        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.request(
                method,
                f"{self.base_url}{path}",
                json=body,
                headers=headers,
            ) as response:
                try:
                    payload = await response.json()
                except Exception:
                    payload = {"error": await response.text()}
                if response.status >= 400:
                    code = payload.get("code", "GREENLAND_GAME_ERROR")
                    message = payload.get("error", str(payload))
                    raise RuntimeError(f"{code}: {message}")
                return payload

    @staticmethod
    def path_value(value: str) -> str:
        return quote(value, safe="")


client = GreenlandGameClient()
