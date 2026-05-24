from fastapi import FastAPI, Request
import tiktoken

app = FastAPI()
encoding = tiktoken.encoding_for_model("gpt-4")

@app.post("/tokenize")
async def tokenize(request: Request):
    data = await request.json()
    text = data.get("text", "")
    tokens = encoding.encode(text)
    return {
        "token_count": len(tokens),
        "tokens": tokens
    }
