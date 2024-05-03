from fastapi import FastAPI
from pydantic import BaseModel
from predict_model import TextPredictor

app = FastAPI()

class Sentence(BaseModel):
    text: str

text_predictor = TextPredictor()

@app.post("/predict/")
async def predict(sentence: Sentence):
    prediction = text_predictor.predict_text(sentence.text)
    return {"original": sentence.text, "classification": prediction}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)