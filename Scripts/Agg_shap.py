from datasets import load_dataset
import shap
from transformers import pipeline, BertTokenizer, BertForSequenceClassification

# Load your custom model and tokenizer
model = BertForSequenceClassification.from_pretrained('/home/j597s263/bert_83').to("cuda")
tokenizer = BertTokenizer.from_pretrained('/home/j597s263/bert_83')

# Initialize the sentiment-analysis pipeline with truncation and GPU support
classifier = pipeline(
            "sentiment-analysis",
                model=model,
                    tokenizer=tokenizer,
                        device=0,  # Use GPU
                            truncation=True  # Enable truncation to limit input length to 512 tokens
                            )

# Load IMDb test dataset
dataset = load_dataset("imdb", split="test")

# Initialize the SHAP explainer
explainer = shap.Explainer(classifier)

# Initialize a dictionary to store aggregated SHAP values for each index
shap_dict = {i: 0.0 for i in range(512)}  # Dictionary with 512 indices initialized to 0.0

# Loop over each example in the dataset
for i, example in enumerate(dataset['text']):
        # Generate SHAP values for the current example
            shap_values = explainer([example])

                # Aggregate SHAP values for each index in the dictionary
                    for idx, shap_value in enumerate(shap_values.values[0]):
                                if idx < 512:  # Ensure we stay within 512 indices
                                                shap_dict[idx] += float(shap_value[0])  # Convert to float before adding

                                                    # Optional: Print progress every 100 examples
                                                        if (i + 1) % 100 == 0:
                                                                    print(f"Processed {i + 1} samples...")

                                                                    # Save the aggregated SHAP values to a text file
                                                                    output_file = "aggregated_shap_values.txt"
                                                                    with open(output_file, "w") as file:
                                                                            for index, value in shap_dict.items():
                                                                                        file.write(f"{index}: {value}\n")

                                                                                        print(f"Aggregated SHAP values saved to {output_file}")

