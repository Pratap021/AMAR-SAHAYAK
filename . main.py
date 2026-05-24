from fastapi import FastAPI, Request
from twilio.twiml.messaging_response import MessagingResponse
from rag_pipeline import load_and_embed_docs, generate_answer

app = FastAPI()
vectorstore = load_and_embed_docs("data/pdfs/ration_card.pdf")

@app.post("/whatsapp")
async def whatsapp_webhook(request: Request):
    form = await request.form()
    user_query = form.get("Body")

    response = MessagingResponse()
    answer = generate_answer(vectorstore, user_query)
    response.message(answer)

    return str(response)
