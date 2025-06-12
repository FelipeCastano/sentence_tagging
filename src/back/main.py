from fastapi import FastAPI
from pydantic import BaseModel
#from model.tagger import tag_text  # Your existing tagging function

app = FastAPI()

class TagRequest(BaseModel):
    text: str
    lang: int  # 1 = English, 2 = Spanish

@app.post("/get_tag")
def get_tag(request: TagRequest):
    ex = {
      "tokens": ["This", "is", "a", "test"],
      "tags": ["DET", "AUX", "DET", "NOUN"]
    }
    #result = tag_text(request.text, request.lang)
    return ex
