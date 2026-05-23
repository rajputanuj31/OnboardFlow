# pyrefly: ignore [missing-import]
from fastapi import FastAPI, HTTPException
from onboard_github import fetch_repo_details, parse_github_repo_url
from models import RepoOnboardRequest, RepoDetails

app = FastAPI(title="OSS Onboarding Agent")


@app.post("/onboard/fetch", response_model=RepoDetails)
async def onboard_fetch(inp: RepoOnboardRequest):
    try:
        repo = parse_github_repo_url(inp.repo_url)
        raw = await fetch_repo_details(repo)
        print(f"\nFetched repository: {raw.repo}")
        print(f"  stars:        {raw.repo_stars}  lang: {raw.repo_language}")
        print(f"  readme:       {'yes (' + str(len(raw.readme)) + ' chars)' if raw.readme else 'not found'}")
        print(f"  contributing: {'yes (' + str(len(raw.contributing)) + ' chars)' if raw.contributing else 'not found'}")
        return raw
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}
