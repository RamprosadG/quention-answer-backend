from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer

# Load the model and tokenizer
model_name = "./model/my_qa_model"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
nlp_qa = pipeline("question-answering", model=model, tokenizer=tokenizer)

def get_answer(question, context=""):
    result = nlp_qa({
        "question": question,
        "context": context
    })
    return result['answer']
