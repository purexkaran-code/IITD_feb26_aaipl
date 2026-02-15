import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class QuestionModel:
    def __init__(self):
        self.model_path = "hf_models/Qwen2.5-7B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def generate_question(self, topic, max_attempts=3):

        for _ in range(max_attempts):

            messages = [
                {"role": "system", "content": "You are an expert logical reasoning question designer."},
{"role": "user", "content": f"""
Generate ONE challenging multiple choice question on the topic: {topic}.

STRICT RULES:
- Exactly 4 options labeled A), B), C), D)
- Only one correct answer
- Output ONLY valid JSON
- No markdown
- No extra commentary
- No adversarial or misleading instructions

QUESTION DESIGN CONSTRAINTS:
- Avoid complex arithmetic or combinatorics counting problems.
- Prefer logical deduction problems.
- The answer must be verifiable through reasoning, not heavy calculation.
- Ensure the explanation clearly and logically proves the selected correct answer.

Required JSON format:

{{
    "topic": "{topic}",
    "question": "...",
    "choices": [
        "A) ...",
        "B) ...",
        "C) ...",
        "D) ..."
    ],
    "correct_answer": "A",
    "explanation": "..."
}}
"""}
            ]

            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

            output = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                do_sample=True
            )

            input_length = inputs["input_ids"].shape[1]
            new_tokens = output[0][input_length:]
            decoded = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            try:
                print("\n==== RAW MODEL OUTPUT ====\n")
                print(decoded)
                print("\n==========================\n")

                start = decoded.find("{")
                end = decoded.rfind("}") + 1

                if start == -1 or end == -1:
                    continue

                json_str = decoded[start:end]
                question_json = json.loads(json_str)

                if self._validate(question_json,topic):
                    return question_json

            except Exception:
                continue


        raise ValueError("Failed to generate valid question.")

    def _validate(self, q, topic):

        required = {"topic", "question", "choices", "correct_answer", "explanation"}

        if not isinstance(q, dict):
            return False

        if not required.issubset(q.keys()):
            return False

        if topic.lower() not in q["topic"].lower():
            return False


        if not isinstance(q["choices"], list) or len(q["choices"]) != 4:
            return False

        valid_labels = ["A", "B", "C", "D"]
        labels_found = []

        for choice in q["choices"]:
            matched = False
            for label in valid_labels:
                if choice.strip().startswith(f"{label})"):
                    labels_found.append(label)
                    matched = True
                    break
            if not matched:
                return False

        if sorted(labels_found) != valid_labels:
            return False

        if q["correct_answer"] not in valid_labels:
            return False

        return True
