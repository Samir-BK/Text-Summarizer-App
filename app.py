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
    text = re.sub(r"<.*?>", " ", text) # all html tags
    text = text.strip().lower()
    return text

# Test the core logic for summarization

def summarize_dialogue(dialogue : str) -> str:
    dialogue = clean_data(dialogue) # clean

    # tokenize
    inputs = tokenizer(
        dialogue,
        max_length = 512,
        padding = "max_length",
        truncation = True,
        return_tensors = "pt"
    ).to(device)

    
    # generate the summary  => token ids
    model.to(device)
    targets = model.generate(
        input_ids = inputs["input_ids"],
        attention_mask = inputs["attention_mask"],
        max_length = 150,
        num_beams = 4, # this will give the best one out of 4 result produced
        early_stopping = True
    )

    # decode the summary, token ids convert to summary
    summary = tokenizer.decode(
        targets[0],
        skip_special_tokens = True,
        clean_up_tokenization_spaces = True
    )

    return summary
