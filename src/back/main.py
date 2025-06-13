from fastapi import FastAPI
from pydantic import BaseModel
from model.viterbi_tagger import ViterbiTagger
from model.utils import assign_unk_english, assign_unk_espanish, load_conllu_data, load_data_english

app = FastAPI()

train_path = "data/WSJ_02-21.pos"
test_path = "data/WSJ_24.pos"
corpus_path = "data/hmm_vocab.txt"
training_corpus, y, vocab = load_data_english(train_path, test_path, corpus_path)

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
