import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from model_final import ParaphraseDetector  # Import your model class

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(model_path="paraphrase-detection/model_final.pth"):
    model = ParaphraseDetector()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def cosine_similarity(tensor1, tensor2):
    return F.cosine_similarity(tensor1, tensor2, dim=0).item()

def predict_paraphrase(model, sentence1, sentence2, tokenizer):
    inputs = tokenizer(
        sentence1, sentence2,
        padding="max_length", truncation=True, max_length=256, return_tensors="pt"
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        logits, encoded1, encoded2 = model(input_ids, attention_mask=attention_mask)
        similarity_score = cosine_similarity(encoded1.squeeze(0)[0], encoded2.squeeze(0)[0])
        prediction = 1 if similarity_score > 0.60 else 0
    return "Paraphrase" if prediction == 1 else "Not a Paraphrase", similarity_score

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = load_model()

    while True:
        sentence1 = input("Enter first sentence: ")
        sentence2 = input("Enter second sentence: ")

        result, score = predict_paraphrase(model, sentence1, sentence2, tokenizer)
        print(f"Result: {result}")
        print(f"Cosine Similarity Score: {score:.4f}\n")
