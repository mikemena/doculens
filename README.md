# DOCULENS

## Activate vm

`source .venv/bin/activate`

## 1. Project Structure

```
doculens/
├── data/
│   ├── raw/                   # your sample .docx, .pdf, .xlsx files
│   └── processed/             # extracted text & metadata (JSON/CSV)
├── src/
│   ├── ingestion/
│   │   ├── pdf_extractor.py   # e.g., pdfplumber or PyMuPDF
│   │   ├── docx_extractor.py  # python-docx
│   │   └── xlsx_extractor.py  # openpyxl / pandas.read_excel
│   ├── preprocessing/
│   │   ├── text_cleaner.py    # remove headers, normalize whitespace, etc.
│   │   └── metadata_builder.py# parse dates/parties via regex or rule-based
│   ├── dataset.py             # assemble a pandas DataFrame of (text, labels)
│   ├── train/
│   │   ├── split.py           # train/test split via sklearn
│   │   └── spaCy_config.cfg   # spaCy pipeline config (NER, textcat, etc.)
│   ├── models/
│   │   └── train_spacy.py     # kicks off spaCy’s training CLI or API
│   ├── inference/
│   │   └── predict.py         # load model & run on new contracts
│   └── utils.py               # shared utility functions
├── requirements.txt
└── README.md
```

## 2. Ingestion & Text Extraction

1. **PDF → text**

   * Use `pdfplumber` or `PyMuPDF` to pull text blocks (and optionally layout info).
2. **DOCX → text**

   * Use `python-docx` to read paragraphs & tables.
3. **Excel → structured data**

   * Excel often holds tabular metadata (e.g., key‐value sheets). Read with `pandas.read_excel` or `openpyxl`.
   * **Note:** spaCy is *not* used for Excel; you extract text or metadata first via pandas.

*Save each document’s full text and any metadata into a JSON or CSV in `data/processed/`.*

## 3. Preprocessing & Labeling

1. **Cleaning** (`src/preprocessing/text_cleaner.py`)

   * Strip line headers/footers, normalize whitespace, remove non-ASCII if needed.
2. **Rule-based metadata** (`metadata_builder.py`)

   * Use regex or spaCy’s [Matcher](https://spacy.io/usage/rule-based-matching) to pull out dates, parties, dollar amounts.
3. **Annotation**

   * For NER or clause classification, prepare your labeled data in spaCy’s JSON format (`docbin`) or as `.spacy` files.

An example of labeling flow (Label Studio is one example)
Raw text → Label Studio → Annotated JSON/CONLL → .spacy files → Config + Training → Trained Model

## 4. Dataset Splitting

* Use `sklearn.model_selection.train_test_split` on your DataFrame of examples (`text, annotations`).
* **Why?** You need reproducible splits—scikit-learn is perfect for that.
* **Not needed:** scaling/normalizing numeric features in the NLP pipeline. Your “features” are token sequences/embeddings, which you don’t manually scale.

```python
from sklearn.model_selection import train_test_split
train, dev = train_test_split(docs, test_size=0.2, random_state=42)
```

## 5. spaCy Pipeline & Training

### Why spaCy?

* **Production-ready**: blazing fast Cython core, easy model packaging.
* **Built-in training** for NER, text categorization, custom components.
* **Config-driven**: one file defines your entire pipeline (tokenizer, components, training hyperparameters).

### High-level flow (`src/train/train_spacy.py`)

1. **Create a `config.cfg`**

   * Specify components you need (e.g., `tokenizer`, `ner`, `textcat`).
   * Point to your train/dev `.spacy` files.
2. **Train**

   ```bash
   python -m spacy train src/train/spacy_config.cfg --output ./models/ --paths.train ./data/train.spacy --paths.dev ./data/dev.spacy
   ```
3. **Evaluate & iterate**

   * spaCy automatically reports precision/recall/F1 on your dev set.

### Custom Neural Layers?

* If you want to build your own PyTorch layers, spaCy lets you plug in a **custom component**:

  ```python
  @Language.factory("my_custom_component")
  def create_component(nlp, name):
      return MyCustomPyTorchComponent(nlp)
  ```
* But for most NER/textcat tasks, spaCy’s built-in architectures (which are already optimized Cython wrappers around Thinc/PyTorch) are more than enough.

## 6. Inference & Deployment

* **Packaged model**: spaCy saves a complete package you can `nlp = spacy.load("path/to/model")`.
* **API**: expose a `/predict` endpoint (FastAPI or Flask) that loads your spaCy model once at startup and serves requests.
* **Dashboard**: hook your API into a simple React or Streamlit front-end to upload a contract PDF and display extracted entities/clauses.

## 7. Monitoring & Maintenance

* **Model drift**: periodically re-annotate new contracts and retrain.
* **Logging**: capture incoming docs, errors, and user feedback to improve your rules and annotations.
* **CI/CD**: add tests for your extraction scripts, spaCy pipeline training, and inference endpoint.

---

**Bottom Line**

1. **Use spaCy** for all your NLP (tokenization, NER, text classification, rule-based matching).
2. **Use scikit-learn** only for dataset splitting (and if you ever add any purely numeric features you do want to scale).
3. **Skip manual scaling/normalization** of text embeddings—spaCy’s pipelines handle those under the hood.
4. **Leverage spaCy’s config-based training** instead of hand-rolling a PyTorch loop, unless you have a very custom architecture in mind.

This flow will keep your code organized, let you iterate quickly on annotations/models, and produce a production-ready pipeline. Good luck!
