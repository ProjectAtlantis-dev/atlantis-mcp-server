import os
from typing import Any, Dict, Optional

import aiohttp


class WarehouseBankClient:
    def __init__(self) -> None:
        self.base_url = os.getenv("GAME_BANK_URL", "http://127.0.0.1:3000/api/bank").rstrip("/")
        self.service_token = os.getenv("WAREHOUSE_SERVICE_TOKEN", "").strip()

    async def request(
        self,
        method: str,
        path: str,
        *,
        body: Optional[Dict[str, Any]] = None,
        service_auth: bool = False,
    ) -> Any:
        headers = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        if service_auth:
            if not self.service_token:
                raise RuntimeError("WAREHOUSE_SERVICE_TOKEN is not configured")
            headers["Authorization"] = f"Bearer {self.service_token}"

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


client = WarehouseBankClient()
