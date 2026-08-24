# Text Summarizer App

A web application that summarizes dialogue and text content using a fine-tuned T5 transformer model, served through a FastAPI backend.

## Overview

This project provides a simple interface for pasting text or dialogue and receiving a concise summary in return. The summarization is powered by a T5 (Text-to-Text Transfer Transformer) model, and the application is built with a FastAPI backend and a lightweight HTML/CSS/JavaScript frontend.

## Features

- Text summarization using a fine-tuned T5 model
- Simple, responsive web interface
- REST API endpoint for programmatic access
- Input cleaning and preprocessing before summarization
- Beam search decoding for improved summary quality

## Tech Stack

**Backend**
- Python
- FastAPI
- Uvicorn (ASGI server)
- Hugging Face Transformers
- PyTorch

**Frontend**
- HTML
- CSS
- Vanilla JavaScript

## Project Structure

```
Text-Summarizer-App/
├── app.py                          # FastAPI application and model logic
├── index.html                      # Frontend page
├── static/
│   ├── script.js                   # Frontend logic (form handling, API calls)
│   └── style.css                   # Styling
├── text-summarizer_saved_summary_model/
│   ├── config.json
│   ├── generation_config.json
│   ├── model.safetensors
│   ├── tokenizer_config.json
│   └── tokenizer.json
├── requirements.txt                # Python dependencies
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.11 or later
- pip

### Installation

1. Clone the repository
   ```bash
   git clone https://github.com/Samir-BK/Text-Summarizer-App.git
   cd Text-Summarizer-App
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application
   ```bash
   python -m uvicorn app:app --reload
   ```

4. Open your browser and navigate to
   ```
   http://127.0.0.1:8000
   ```

## API Usage

### `POST /summarize/`

Summarizes the provided text.

**Request body**
```json
{
  "dialogue": "Your text or dialogue to summarize goes here."
}
```

**Response**
```json
{
  "summary": "Generated summary text."
}
```

## How It Works

1. Input text is cleaned (whitespace normalization, HTML tag removal, lowercasing).
2. The cleaned text is tokenized and passed to the T5 model.
3. The model generates a summary using beam search decoding.
4. The generated token sequence is decoded back into readable text and returned to the frontend.

## Notes

- The model files are loaded locally from the `text-summarizer_saved_summary_model/` directory.
- `torch` will automatically use GPU (CUDA or MPS) if available, otherwise it falls back to CPU.

## License

This project is provided as-is for educational and demonstration purposes.
