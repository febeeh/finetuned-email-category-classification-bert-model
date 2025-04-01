# Fine-Tuned Email Category Classification using BERT (RoBERTa)
This repository provides a **fine-tuned** **BERT (RoBERTa)** model for classifying emails into three categories:  
- 🏷 **Promotions** (Marketing emails, discount offers, etc.)  
- 🔔 **Updates** (Account notifications, newsletters, etc.)  
- 👥 **Social** (Friend requests, social media notifications, etc.)  

## 🎯 Fine-Tuning Details  
- **Model:** RoBERTa (Fine-tuned on a personal email dataset)  
- **Dataset:** Custom-labeled real-world emails  
- **Fine-Tuned Using:** Hugging Face Transformers & PyTorch  

### Steps to Use the Model:
#### 1: Download model file - [Click here to download](https://www.playbook.com/s/febeeh/2hwJjcHJDLE8kWnssJZoWbf9?assetToken=zYxx8BTcRaviaFKDrtjxVJ1j)
#### 2: Extract model into project folder ```(email_classifier_roberta_model/)```
#### 1: Load the trained model and vectorizer
```python
model = RobertaForSequenceClassification.from_pretrained("email_classifier_roberta_model")
tokenizer = RobertaTokenizer.from_pretrained("email_classifier_roberta_model")
```
#### 2: Prepare your input text
```python
sampleMail = "You have a new friend request from facebook."
```
#### 3: Make a prediction
```python
inputs = tokenizer(sampleMail, return_tensors="pt", padding=True, truncation=True, max_length=128)
outputs = model(**inputs)
predicted_class = torch.argmax(outputs.logits, dim=1).item()
```
#### 4: Decode the predicted class
```python
label = {0: "Promotions", 1: "Updates", 2: "Social"}[predicted_class]
print(f"Predicted Category: {label}")
```

### Executing the Jupyter Notebook
The provided Jupyter Notebook ```(mail_category_classification.ipynb)``` contains step-by-step instructions to test the model. Follow these steps to execute it:
#### 1: Open the notebook:
```jupyter notebook mail_category_classification.ipynb```
#### 2: Run all cell

______________________________________________

## Done
