import os
from typing import Any, Dict, Optional

import aiohttp


class GameBankClient:
    """HTTP adapter used only behind the Atlantis dynamic-function interface."""

    @property
    def base_url(self):
        return os.getenv("GAME_BANK_URL", "").rstrip("/")

    @property
    def authority_token(self):
        return os.getenv("GAME_BANK_AUTHORITY_TOKEN", "").strip()

    async def request(
        self,
        method: str,
        path: str,
        *,
        body: Optional[Dict[str, Any]] = None,
        authority: bool = False,
    ) -> Any:
        if not self.base_url:
            raise RuntimeError('GAME_BANK_URL must explicitly select the bank service')
        headers = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        if authority or self.authority_token:
            if not self.authority_token:
                raise RuntimeError(
                    "GAME_BANK_AUTHORITY_TOKEN is not configured on the central bank MCP service"
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
                    code = payload.get("code", "GAME_BANK_ERROR") if isinstance(payload, dict) else "GAME_BANK_ERROR"
                    message = payload.get("error", str(payload)) if isinstance(payload, dict) else str(payload)
                    raise RuntimeError(f"{code}: {message}")
                return payload


client = GameBankClient()
