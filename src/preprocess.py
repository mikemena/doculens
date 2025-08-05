import spacy
from sklearn.model_selection import train_test_split

nlp = spacy.blank("en")


def clean_text(text):
    text = text.replace("\n", " ").strip()
    doc = nlp(text)  # tokenize with spaCy
    return " ".join(
        token.text.lower() for token in doc if not token.is_stop and not token.is_punct
    )


# If training: gather texts, split
texts = [extract_text(f) for f in os.listdir("data/") if f.endswith((".pdf", ".docx"))]
train_texts, test_texts = train_test_split(texts, test_size=0.2)
