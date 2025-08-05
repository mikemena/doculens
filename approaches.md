## 1. Named Entity Recognition (NER)

**Overview**
Detect and classify entities (e.g. PERSON, ORG) at the token level.

**Preprocessing**

* **Tokenize & POS-tag** (spaCy)

  ```python
  import spacy
  nlp = spacy.load("en_core_web_sm")
  doc = nlp("Acme Corp. was founded on 01/01/2025.")
  for token in doc:
      print(token.text, token.pos_)
  ```
* **Clean OCR noise** (regex or custom rules)
* **Integrate gazetteers** (load known names/dates)

**Modeling**

* **Transformer fine-tuning** (Hugging Face)

  ```python
  from transformers import BertForTokenClassification
  model = BertForTokenClassification.from_pretrained("bert-base-cased", num_labels=9)
  ```
* **BiLSTM-CRF** (flair or AllenNLP)

---

## 2. Text Classification

### 2.1 Document-Level

**Overview**
Assign labels to entire documents (e.g. “contract”, “news article”).

**Preprocessing**

* **Clean & vectorize**

  ```python
  from sklearn.feature_extraction.text import TfidfVectorizer
  vect = TfidfVectorizer(max_features=5000)
  X = vect.fit_transform(docs)
  ```
* **Truncate/summary** to fit token limits

**Modeling**

* **BERT sequence classification**

  ```python
  from transformers import BertForSequenceClassification
  model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
  ```
* **Hierarchical Attention Networks** (e.g. with keras-han)

### 2.2 Paragraph/Section-Level

**Overview**
Label sections independently (e.g. “payment terms”, “termination clause”).

**Preprocessing**

* **Split by headings**

  ```python
  import re
  sections = re.split(r"\n#+\s", document)
  ```

**Modeling**

* **Longformer classification**

  ```python
  from transformers import LongformerForSequenceClassification
  model = LongformerForSequenceClassification.from_pretrained("allenai/longformer-base-4096")
  ```

### 2.3 Sentence-Level

**Overview**
Classify individual sentences (e.g. sentiment, fact/opinion).

**Preprocessing**

* **Sentence tokenize** (NLTK)

  ```python
  from nltk import sent_tokenize
  sentences = sent_tokenize(text)
  ```

**Modeling**

* **Lightweight RNN/CNN** (PyTorch)

  ```python
  import torch.nn as nn
  rnn = nn.GRU(input_size=300, hidden_size=128, batch_first=True)
  ```
* **BERT sequence classification**

---

## 3. Information Extraction

### 3.1 Relation Extraction

**Overview**
Identify semantic relationships between entity pairs.

**Preprocessing**

* **Dependency parse** (spaCy)

  ```python
  for token in doc:
      print(token.text, token.dep_, token.head.text)
  ```
* **Coreference resolution** (neuralcoref)

**Modeling**

* **Graph Convolutional Networks** (PyTorch Geometric)

  ```python
  from torch_geometric.nn import GCNConv
  conv = GCNConv(in_channels, out_channels)
  ```
* **Transformer with entity markers**

### 3.2 Event Extraction

**Overview**
Detect events (triggers + arguments) in text.

**Preprocessing**

* **Trigger identification** (HeidelTime for time triggers)
* **Temporal normalization**

**Modeling**

* **Pipeline**: BERT for trigger + sequence labeler for roles
* **End-to-end**: DyGIE++ (AllenAI)

---

## 4. Document Structure Analysis

### 4.1 Layout Analysis

**Overview**
Detect visual elements (tables, figures, headers).

**Preprocessing**

* **OCR + image cleanup** (Tesseract + OpenCV)

  ```bash
  tesseract page.png stdout --dpi 300
  ```

**Modeling**

* **LayoutLM** (Hugging Face)

  ```python
  from transformers import LayoutLMForTokenClassification
  model = LayoutLMForTokenClassification.from_pretrained("microsoft/layoutlm-base-uncased")
  ```
* **Faster R-CNN** for element detection

### 4.2 Logical Structure Analysis

**Overview**
Build hierarchical section/heading tree.

**Preprocessing**

* **Font/size heuristics** (PDFMiner)
* **Tree construction** via TOC parsing

**Modeling**

* **Hi-Transformer** for hierarchical encoding
* **Graph Neural Networks** on section nodes

---

## 5. Content Understanding

### 5.1 Summarization

**Overview**
Generate extractive or abstractive summaries.

**Preprocessing**

* **TextRank extractive** (Gensim)

  ```python
  from gensim.summarization import summarize
  summary = summarize(text, ratio=0.2)
  ```

**Modeling**

* **BART abstractive**

  ```python
  from transformers import BartForConditionalGeneration
  model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
  ```
* **T5 summarization**

### 5.2 Question Answering

**Overview**
Answer questions using document context.

**Preprocessing**

* **Dense retrieval** (SentenceTransformers + FAISS)

**Modeling**

* **BERT extractive**

  ```python
  from transformers import BertForQuestionAnswering
  model = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
  ```

### 5.3 Topic Modeling

**Overview**
Discover latent topics in a corpus.

**Preprocessing**

* **Bag-of-words + TF-IDF**

**Modeling**

* **LDA** (Gensim)

  ```python
  from gensim.models.ldamodel import LdaModel
  lda = LdaModel(corpus, num_topics=10, id2word=dictionary)
  ```
* **BERTopic**

---

## 6. Specialized Analysis Types

### 6.1 Key-Value Pair Extraction

**Overview**
Extract form fields (e.g. “Name: John Doe”).

* **Library**: LayoutLMv3, FUNSD dataset

### 6.2 Table Extraction & Understanding

**Overview**
Detect and parse tabular data.

* **Tool**: Camelot

  ```python
  import camelot
  tables = camelot.read_pdf("doc.pdf")
  ```

### 6.3 Signature & Handwriting Analysis

**Overview**
Verify signatures or transcribe handwriting.

* **Library**: OpenCV + Siamese CNNs

### 6.4 Comparison & Difference Analysis

**Overview**
Detect edits/changes between document versions.

* **Library**: difflib

  ```python
  import difflib
  diffs = difflib.ndiff(text1.splitlines(), text2.splitlines())
  ```

---

## 7. Advanced Semantic Analysis

### 7.1 Semantic Search

* **Library**: Sentence-Transformers + FAISS

### 7.2 Intent Recognition

* **Framework**: Rasa NLU or Hugging Face intent classifier

### 7.3 Sentiment & Emotion Analysis

* **Model**: cardiffnlp/twitter-roberta-base-emotion

---

## 8. Compliance & Risk Analysis

### 8.1 Compliance Checking

* **Approach**: Zero-shot with DeBERTa via Hugging Face pipeline

### 8.2 Risk Assessment

* **Model**: BERT + regression head (PyTorch)

---

## 9. Evaluation Metrics & Monitoring

* **NER**: `seqeval` for entity-level F1
* **Summarization**: `rouge-score`
* **QA**: SQuAD metrics (Exact Match, F1)

---

## 10. Data Augmentation & Domain Adaptation

* **Back-translation**: nlpaug
* **Layout augment**: OpenCV rotations
* **Continual pretraining**: Hugging Face `run_mlm.py`

---
