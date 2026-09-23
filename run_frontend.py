"""Start the local registration and identification web app."""
from __future__ import annotations

import uvicorn


if __name__ == "__main__":
    uvicorn.run("noseid.api.app:app", host="127.0.0.1", port=8000, reload=False)

