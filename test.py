import torch
from transformers import BertTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(model_path="paraphrase-detection\model_final.pth"):

    model = torch.load(model_path, map_location=device)
    model.to(device)
    model.eval()
    return model

def predict_paraphrase(model, sentence1, sentence2, tokenizer):
    inputs = tokenizer(
        sentence1, sentence2,
        padding="max_length", truncation=True, max_length=256, return_tensors="pt"
    )
    
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        prediction = torch.argmax(outputs, dim=1).item()
    
    return "Paraphrase" if prediction == 1 else "Not a Paraphrase"

if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = load_model()
    
    while True:
        sentence1 = input("Enter first sentence: ")
        sentence2 = input("Enter second sentence: ")
        
        result = predict_paraphrase(model, sentence1, sentence2, tokenizer)
        print(f"Result: {result}\n")
