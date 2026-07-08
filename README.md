# 🤖 Deep Space Station 9 AI Assistant

NLP-powered FAQ chatbot for astronaut operations. Matches free-form questions to 52 known FAQs using **TF-IDF vectorization + cosine similarity** — entirely local, **no external API, no API key, no internet dependency** for the AI logic itself.

## Why no Claude/OpenAI/Google API key?
This is a classic, lightweight retrieval-based chatbot. It doesn't need an LLM call at all — scikit-learn's TF-IDF + cosine similarity is the standard, free, and fully offline approach for FAQ matching, and it runs instantly with no rate limits or cost.

## Features
- 52 FAQs across 7 categories (Operations, Emergency, Communications, Medical, Alien Contact, Navigation, General)
- TF-IDF + cosine similarity matching with confidence scoring (High/Medium/Low)
- Emergency keyword detection with visual alert styling
- Related question suggestions
- Searchable FAQ sidebar (click any FAQ to ask it)
- Chat history with clear button
- Fully responsive space-themed UI

## Setup

```bash
cd task2-deep-space-ai-assistant
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

First run automatically downloads required NLTK data (`punkt`, `stopwords`). If your environment has no internet access at all, the app falls back to a built-in stopword list automatically — no crash either way.

Open **http://localhost:5000**

## Project Structure
```
task2-deep-space-ai-assistant/
├── app.py
├── requirements.txt
├── templates/
│   └── chat.html
├── static/
│   ├── css/style.css
│   └── js/main.js
└── README.md
```

## Tech Stack
- Backend: Flask
- NLP: scikit-learn (TfidfVectorizer, cosine_similarity), NLTK
- Frontend: HTML/CSS/JavaScript, no frameworks

## How the matching works
1. User question is lowercased, punctuation stripped, tokenized, stopwords removed
2. Converted into a TF-IDF vector using the same vectorizer fit on all 52 FAQ questions
3. Cosine similarity computed against all FAQ vectors
4. Best match returned with a confidence percentage; below 15% similarity, the bot admits uncertainty instead of guessing

## License
MIT
