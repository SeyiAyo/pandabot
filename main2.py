#This code is a comaprative code by chatgpt
#Not fot Prod
import json
import torch
from transformers import BertTokenizer, BertForQuestionAnswering
from difflib import get_close_matches


def find_best_match(user_question: str, questions: List[str]) -> Union[str, None]:
    matches: list = get_close_matches(user_question, questions, n=1, cutoff=0.6)
    return matches[0] if matches else None

def load_knowledge_base(file_path: str) -> dict:
    with open(file_path, 'r') as file:
        data: dict = json.load(file)
    return data

def save_knowledge_base(file_path :str, data: dict):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=2)

def get_answer_for_question(question: str, model: BertForQuestionAnswering, tokenizer: BertTokenizer) -> str | None:
    encoding = tokenizer(question, return_tensors="pt")
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
        
    with torch.no_grad():
        start_scores, end_scores = model(input_ids, attention_mask=attention_mask)

    start_idx = torch.argmax(start_scores, dim=1)
    end_idx = torch.argmax(end_scores, dim=1) + 1

    answer_tokens = input_ids[0][start_idx:end_idx]
    answer = tokenizer.decode(answer_tokens)

    return answer

def chat_bot():
    knowledge_base: dict = load_knowledge_base('knowledge_base.json')

    while True:
        user_input: str = input('You: ')

        if user_input.lower() == 'quit':
            break

        best_match: str | None = find_best_match(user_input, [q["question"] for q in knowledge_base["questions"]])

        if best_match:
            answer: str = get_answer_for_question(best_match, model, tokenizer)
            print(f'PandaBot: {answer}')
        else:
            print('PandaBot: I don\'t know the answer. Can you teach me?')
            new_answer: str = input('Type the answer or "skip" to skip: ')

            if new_answer.lower() != 'skip':
                knowledge_base["questions"].append({"question": user_input, "answer": new_answer})
                save_knowledge_base('knowledge_base.json', knowledge_base)
                print('PandaBot: Thank You! I learned a new response!')

def chat_bot():
    knowledge_base: dict = load_knowledge_base('knowledge_base.json')

    while True:
        user_input: str = input('You: ')

        if user_input.lower() == 'quit':
            break

        best_match: str | None = find_best_match(user_input, [q["question"] for q in knowledge_base["questions"]])

        if best_match:
            answer: str = get_answer_for_question(best_match, model, tokenizer)
            print(f'PandaBot: {answer}')
        else:
            print('PandaBot: I don\'t know the answer. Can you teach me?')
            new_answer: str = input('Type the answer or "skip" to skip: ')

            if new_answer.lower() != 'skip':
                knowledge_base["questions"].append({"question": user_input, "answer": new_answer})
                save_knowledge_base('knowledge_base.json', knowledge_base)
                print('PandaBot: Thank You! I learned a new response!')
