import spacy
from spacy.matcher import PhraseMatcher
from logger import setup_logger
from ingestion import extract_text

logger = setup_logger(__name__, include_location=True)

nlp = spacy.load("en_core_web_sm")
matcher = PhraseMatcher(nlp.vocab)
risk_terms = [nlp("missing clause"), nlp("high risk")]
matcher.add("RISK", risk_terms)


def analyze_contract(text):
    doc = nlp(text)
    entities = {(ent.text, ent.label_) for ent in doc.ents}  # Extract NER
    risks = [doc[start:end].text for match_id, start, end in matcher(doc)]  # Flag risks
    return {"entities": entities, "risks": risks}


if __name__ == "__main__":
    logger.info("\n" + "=" * 70)
    logger.info("Starting train model...")
    results = analyze_contract(extract_text("../data/raw/Service Agreement 1.docx"))

    logger.info(f"text: {results}")
