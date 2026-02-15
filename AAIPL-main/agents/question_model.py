import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class QuestionModel:
    def __init__(self):

        model_path = "hf_models/Qwen2.5-7B-Instruct"

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto"
        )

        self.model.eval()

    def generate_question(self, topic):

        prompt = f"""
You are an expert logical reasoning question designer.

Generate ONE multiple choice question.

TOPIC: {topic}

RULES:
- Logical reasoning only
- Avoid heavy calculations
- Exactly 4 options
- Only one correct answer
- Return ONLY valid JSON

FORMAT:
{{
 "topic": "{topic}",
 "question": "...",
 "choices": ["A) ...","B) ...","C) ...","D) ..."],
 "answer": "A",
 "explanation": "brief explanation"
}}
"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        output = self.model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1
        )

        text = self.tokenizer.decode(output[0], skip_special_tokens=True)

        start = text.find("{")
        end = text.rfind("}") + 1

        return json.loads(text[start:end])
if __name__ == "__main__":
    qm = QuestionModel()
    question = qm.generate_question("Probability")
    print(json.dumps(question, indent=2))