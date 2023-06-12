---
language:
- ar
- ca
- de
- el
- en
- es
- fr
- hi
- it
- ja
- ko
- nl
- pl
- pt
- ru
- sv
- vi
- zh
widget:
- text: >-
    The Red Hot Chili Peppers were formed in Los Angeles by Kiedis, Flea, guitarist Hillel Slovak and drummer Jack Irons.
tags:
- seq2seq
- relation-extraction

license: cc-by-nc-sa-4.0
---
# RED<sup>FM</sup>: a Filtered and Multilingual Relation Extraction Dataset

This a multilingual version of [REBEL](https://huggingface.co/Babelscape/rebel-large). It can be used as a standalone multulingual Relation Extraction system, or as a pretrained system to be tuned on multilingual Relation Extraction datasets.

mREBEL is introduced in the ACL 2023 paper [RED^{FM}: a Filtered and Multilingual Relation Extraction Dataset](https://github.com/Babelscape/rebel/blob/main/docs/). We present a new multilingual Relation Extraction dataset and train a multilingual version of REBEL which reframed Relation Extraction as a seq2seq task. The paper can be found [here](https://github.com/Babelscape/rebel/blob/main/docs/). If you use the code or model, please reference this work in your paper:

    @inproceedings{huguet-cabot-et-al-2023-red,
        title = "RED^{FM}: a Filtered and Multilingual Relation Extraction Dataset",
        author = "Huguet Cabot, Pere-Llu{\'\i}s  and
          Navigli, Roberto",
        booktitle = "ACL 2023",
        month = jul,
        year = "2023",
        address = "Toronto, Canada",
        publisher = "Association for Computational Linguistics",
    }

The original repository for the paper can be found [here](https://github.com/Babelscape/rebel)

Be aware that the inference widget at the right does not output special tokens, which are necessary to distinguish the subject, object and relation types. For a demo of REBEL and its pre-training dataset check the [Spaces demo](https://huggingface.co/spaces/Babelscape/rebel-demo).

## Pipeline usage

```python
from transformers import pipeline

triplet_extractor = pipeline('text2text-generation', model='Babelscape/mrebel-large', tokenizer='Babelscape/mrebel-large')
# We need to use the tokenizer manually since we need special tokens.
extracted_text = triplet_extractor.tokenizer.batch_decode([triplet_extractor("The Red Hot Chili Peppers were formed in Los Angeles by Kiedis, Flea, guitarist Hillel Slovak and drummer Jack Irons.", return_tensors=True, return_text=False)[0]["generated_token_ids"]])
print(extracted_text[0])
# Function to parse the generated text and extract the triplets
def extract_triplets(text):
    triplets = []
    relation, subject, relation, object_ = '', '', '', ''
    text = text.strip()
    current = 'x'
    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '':
        triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
    return triplets
extracted_triplets = extract_triplets(extracted_text[0])
print(extracted_triplets)
```

## Model and Tokenizer using transformers

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def extract_triplets(text):
    triplets = []
    relation, subject, relation, object_ = '', '', '', ''
    text = text.strip()
    current = 'x'
    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '':
        triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
    return triplets

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")
gen_kwargs = {
    "max_length": 256,
    "length_penalty": 0,
    "num_beams": 3,
    "num_return_sequences": 3,
}

# Text to extract triplets from
text = 'The Red Hot Chili Peppers were formed in Los Angeles by Kiedis, Flea, guitarist Hillel Slovak and drummer Jack Irons.'

# Tokenizer text
model_inputs = tokenizer(text, max_length=256, padding=True, truncation=True, return_tensors = 'pt')

# Generate
generated_tokens = model.generate(
    model_inputs["input_ids"].to(model.device),
    attention_mask=model_inputs["attention_mask"].to(model.device),
    **gen_kwargs,
)

# Extract text
decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)

# Extract triplets
for idx, sentence in enumerate(decoded_preds):
    print(f'Prediction triplets sentence {idx}')
    print(extract_triplets(sentence))
```