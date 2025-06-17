from transformers import AutoTokenizer, AutoModelForSequenceClassification
tok = AutoTokenizer.from_pretrained('distilbert-base-uncased')
clf = AutoModelForSequenceClassification.from_pretrained('data/models/policy')

def select_strategy(state:str) -> str:
    logits = clf(**tok(state, return_tensor='pt')).logits
    return 'persuade' if logits.softmax(-1)[0, 1] > .5 else "challenge"