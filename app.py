from fastapi import FastAPI, Requests
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re
from fastapi.templating import Jinja2Templates #UI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# initialzie our fast api
app = FastAPI(title = "Text Summarizer App", description = "Text Summarization using T5", version = "1.0")

# model and tokenizer
model = T5ForConditionalGeneration.from_pretrained("D:\Text Summarizer Project\Text-Summarizer-App\text-summarizer_saved_summary_model")
tokenizer = T5Tokenizer.from_pretrained("D:\Text Summarizer Project\Text-Summarizer-App\text-summarizer_saved_summary_model")

# device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model.to(device)

# templating
templates = Jinja2Templates(directory = ".")

# Input schema for dailogue => format defining 
class DialogueInput(BaseModel):
    dialogue: str

# clean data funcx
def clean_data(text):
    text = re.sub(r"\r\n", " ", text) # lines
    text = re.sub(r"\s+", " ", text) # spaces
    text = re.sub(r"<.*?>", " ", text) # html tags
    text = text.strip().lower()
    return text