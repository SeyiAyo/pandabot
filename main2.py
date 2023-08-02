!pip install torch json difflib
!pip install transformers
import torch
import json
from difflib import get_close_matches
from transformers import BertTokenizer, BertForQuestionAnswering
from typing import List, Union

#Load the knowledgebase from JSON file
def load_knowledge_base(file_path: str) -> dict:
    with open(file_path, 'r') as file:
        data: dict = json.load(file)
    return data

def save_knowledge_base(file_path :str, data: dict):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=2)

def find_best_match(user_question: str, questions: List[str]) -> Union[str, None]:
    matches: list = get_close_matches(user_question, questions, n=1, cutoff=0.6)
    return matches[0] if matches else None

def get_answer_for_question(question: str, knowledge_base: dict, model, tokenizer) -> Union[str, None]:
    for q in knowledge_base["questions"]:
        if q["question"] == question:
            return q["answer"]

    # If the question is not found in the knowledge base, use the NLP model
    inputs = tokenizer.encode_plus(question, return_tensors="pt", add_special_tokens=True, pad_to_max_length=True)
    input_ids = inputs["input_ids"].tolist()[0]

    with torch.no_grad():
        start_scores, end_scores = model(**inputs)

    start_idx = torch.argmax(start_scores)
    end_idx = torch.argmax(end_scores)

    answer = tokenizer.decode(input_ids[start_idx:end_idx+1], skip_special_tokens=True)
    return answer if answer else None

def chat_bot():
    knowledge_base: dict = load_knowledge_base('knowledge_base.json')
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")
    while True:
        user_input: str = input('You: ')

        if user_input.lower() == 'quit':
            save_knowledge_base('knowledge_base.json', knowledge_base)
            break

        best_match: str | None = find_best_match(user_input, [q["question"] for q in knowledge_base["questions"]])

        if best_match:
            answer: str = get_answer_for_question(best_match, knowledge_base, model, tokenizer)
            print(f'PandaBot: {answer}')
        else:
            print('PandaBot: I don\'t know the answer. Let me check...')
            inputs = tokenizer.encode_plus(user_input, return_tensors="pt")
            with torch.no_grad():
                start_scores, end_scores = model(**inputs, return_dict=False) 

            start_idx = torch.argmax(start_scores)
            end_idx = torch.argmax(end_scores)

            generated_answer = tokenizer.decode(inputs["input_ids"][0, start_idx:end_idx+1])

            print(f'PandaBot: Is this the correct answer? (yes/no)')
            print(f'Generated Answer: {generated_answer}')

            confirmation = input('You: ')
            if confirmation.lower() == 'yes':
                knowledge_base["questions"].append({"question": user_input, "answer": generated_answer})
                save_knowledge_base('knowledge_base.json', knowledge_base)
                print('PandaBot: Thank You! I learned a new response!')
            else:
                print('PandaBot: I\'m sorry, please provide the correct answer.')
    
if __name__ == '__main__':
    chat_bot()