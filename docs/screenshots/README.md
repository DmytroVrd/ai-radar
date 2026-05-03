    # Screenshots

Place project screenshots in this folder before adding them to the main README.

Recommended files:

- `landing-page.png`: `http://localhost:8000` showing the AI Radar landing page.
- `query.png`: `http://localhost:8000/docs` with `POST /query` expanded and a successful answer with sources visible.
- `stats.png`: `GET /stats` showing indexed documents.
- `index.png`: successful `POST /index` response.
- `health.png`: successful `GET /health` response.

Capture tips:

- Start Qdrant first: `docker compose up qdrant -d`.
- Start the API: `python main.py`.
- Use a normal browser zoom level, ideally 100%.
- Do not include `.env`, API keys, terminal secrets, or browser extensions with sensitive information.
- Prefer 1280px or wider screenshots so README images stay readable.

After adding screenshots, reference them from the root README like:

```markdown
![AI Radar landing page](docs/screenshots/landing-page.png)
```
