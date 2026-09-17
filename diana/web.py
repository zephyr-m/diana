"""Mount the personal console before runner's legacy root redirect."""
from pathlib import Path
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


def install(app):
    assets = Path(__file__).resolve().parents[1] / "web"

    @app.get("/", include_in_schema=False)
    async def console():
        return FileResponse(assets / "index.html", headers={"Cache-Control": "no-store"})

    app.mount("/console", StaticFiles(directory=assets), name="console")
