# Module template

Copy this folder to `app/modules/<your-module>/` and rename references.
Then add one line in `app/api/v1/router.py`:

```python
from app.modules.<your_module>.router import router as <your_module>_router
api_v1.include_router(<your_module>_router, prefix="/<your-module>", tags=["<your-module>"])
```

See `docs/ADDING_A_MODULE.md` for the full walkthrough.
