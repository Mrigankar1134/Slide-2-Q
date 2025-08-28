# PPTX Question Generator

React + FastAPI app that uploads a PPTX and generates varied questions from each slide using TF-IDF keywords, optional LDA topics, and domain rules.

## Setup

- Models: place `tfidf_vectorizer.pkl` and `lda_model.pkl` in `backend/models/` (already moved if present).

### 1) Install dependencies

```bash
npm run install:all
```

If the spaCy model fails to download, the app will still work with a minimal pipeline (NER-driven rules will be less accurate).

### 2) Run both servers

```bash
npm run dev
```

- Backend: http://localhost:8000
- Frontend: http://localhost:5173

## API

- POST `/generate-questions` with `multipart/form-data` file field `file` containing a `.pptx`.
- Response: `{ slides: [{ slide_index, text, keywords, topics, questions: [{ type, question }] }] }`

## Notes

- Sentence splitting is regex-based for portability. If you want, enable spaCy sentencizer.
- If LDA/TF-IDF models are missing, the app still generates questions using rules.
