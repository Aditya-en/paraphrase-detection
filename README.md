<h1 align="center">Turing's Playground</h1>
<p align="center">
</p>
<a href="https://weekendofcode.computercodingclub.in/"> 
    <img src="https://i.postimg.cc/njCM24kx/woc.jpg" height=30px> 
</a>

## Introduction:
Turing's Playground is an advanced AI-based system designed to detect paraphrases between two given sentences. The model is built using a **Transformer-based architecture** and trained on a **paraphrase detection dataset**. It aims to enhance natural language understanding by accurately determining whether two sentences convey the same meaning. The system predicts a binary output (**Paraphrase / Not a Paraphrase**) along with a **cosine similarity score** between the encoded representations of both sentences.

---

## Table of Contents:
1. [Technology Stack](#technology-stack)  
2. [Dataset](#dataset)  
3. [Model Architecture](#model-architecture)  
4. [Training Process](#training-process)  
5. [Testing & Evaluation](#testing--evaluation)  
6. [Contributors](#contributors)  
7. [Training Loss](#training-loss)  
8. [Made At](#made-at)  

---

## Technology Stack:
The project is built using the following technologies:

- **Python** – Core programming language  
- **PyTorch** – Deep learning framework  
- **Transformers (Hugging Face)** – Pretrained embeddings & tokenization  
- **Jupyter Notebook** – Model training & experimentation  

---

## Dataset:
The model is trained on a **paraphrase detection dataset**, where each data sample consists of:  
- **Sentence 1**
- **Sentence 2**
- **Label (1 = Paraphrase, 0 = Not a Paraphrase)**  

The dataset is tokenized using **BERT-base-uncased** and split into **training and validation sets**.

---

## Model Architecture:
The paraphrase detection model is based on a **Transformer Encoder** architecture with additional layers:

- **Pretrained Transformer Encoder**: Extracts contextual embeddings from sentences.  
- **Multi-Head Attention Layers**: Enhances interactions between sentence pairs.  
- **Fully Connected Layers**: Combines sentence representations to make predictions.  
- **Cosine Similarity Calculation**: Measures semantic closeness between sentence embeddings.  
![Model architecture](paraphrase-detection/BERT_NLP_model_architecture_d285530efe.webp)
---


## Training Process:
The model is trained using **cross-entropy loss** for classification, while cosine similarity is used as an additional metric for sentence embeddings.  

### **Hyperparameters:**
- **Pretrained Model:** `bert-base-uncased`
- **Batch Size:** 32  
- **Learning Rate:** `3e-5` (with warm-up schedule)  
- **Optimizer:** AdamW  
- **Dropout Rate:** 0.45  

### **Training Loss Graph:**
The following graph represents the training and validation loss over multiple epochs:

![Training Loss](paraphrase-detection/loss_plot.png)

---

## Testing & Evaluation:
After training, the model is tested using a separate validation set. The evaluation involves:

- **Binary Classification Accuracy**  
- **Precision, Recall, F1 Score**  
- **Cosine Similarity Score for Sentences**  

To test the model manually, you can use the `test.py` script by providing two sentences as input.

---

## Contributors:
**Team Name:** B200

- [Aditya Sahani](https://github.com/Aditya-en)  
- [Krishna Mohan](https://github.com/kmohan321)  

---

## Made at:
<a href="https://weekendofcode.computercodingclub.in/"> 
    <img src="https://i.postimg.cc/mrCCnTbN/tpg.jpg" height=30px> 
</a>
