from transformers import AutoModelForQuestionAnswering, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd

# Load your dataset
data = pd.read_csv("data/qa_data.csv")

# Convert to a Hugging Face dataset format
train_data = [{"context": row['Context'], "question": row['Question'], "answers": {"text": row['Answer']}} for index, row in data.iterrows()]
dataset = Dataset.from_pandas(pd.DataFrame(train_data))

# Split dataset into train and validation sets
train_test_split = dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

# Load a pre-trained model and tokenizer
model_name = "distilbert-base-uncased-distilled-squad"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Prepare the dataset (tokenize)
def preprocess_function(examples):
    return tokenizer(examples['question'], examples['context'], truncation=True, padding='max_length', max_length=512)

train_dataset = train_dataset.map(preprocess_function, batched=True)
eval_dataset = eval_dataset.map(preprocess_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./model/results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained("./model/my_qa_model")
tokenizer.save_pretrained("./model/my_qa_model")
