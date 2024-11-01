from datasets import load_dataset
import shap
from transformers import pipeline, BertTokenizer, BertForSequenceClassification


model = BertForSequenceClassification.from_pretrained('/home/j597s263/bert_83').to("cuda")
tokenizer = BertTokenizer.from_pretrained('/home/j597s263/bert_83')

classifier = pipeline(
    "sentiment-analysis",
    model=model,
    tokenizer=tokenizer,
    device=0,  
    truncation=True 
)


dataset = load_dataset("imdb", split="test")

explainer = shap.Explainer(classifier)

shap_dict = {i: 0.0 for i in range(512)}  

for i, example in enumerate(dataset['text']):
    shap_values = explainer([example])

    for idx, shap_value in enumerate(shap_values.values[0]):
        if idx < 512:  
            shap_dict[idx] += float(shap_value[0]) 

    
    if (i + 1) % 100 == 0:
        print(f"Processed {i + 1} samples...")

output_file = "ValueMap.txt"
with open(output_file, "w") as file:
    for index, value in shap_dict.items():
        file.write(f"{index}: {value}\n")

print(f"Aggregated SHAP values saved to {output_file}")
