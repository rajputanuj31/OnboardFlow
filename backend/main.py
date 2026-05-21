from fastapi import FastAPI, HTTPException
from issue_finder import IssueFinder
from models import UserInput, IssueSearchResult

app = FastAPI(title="OSS Contributor Agent")
issue_finder = IssueFinder()


@app.post("/issues", response_model=IssueSearchResult)
async def find_issues(user_input: UserInput):
    try:
        return await issue_finder.find(user_input)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}
