import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class AnswerModel:
    def __init__(self):
        self.model_path = "hf_models/Qwen2.5-7B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def answer_question(self, question_input):

        # If input is JSON dict
        if isinstance(question_input, dict):
            question_text = f"""
Question:
{question_input.get("question", "")}

Choices:
{chr(10).join(question_input.get("choices", []))}
"""
        else:
            # If input is plain text
            question_text = str(question_input)

        prompt = f"""
You are a highly careful logical reasoning expert.

Read the question carefully.
Think silently.
Select the single correct option.

Return ONLY one letter: A, B, C, or D.

{question_text}
"""

        messages = [
            {"role": "system", "content": "You are a precise logical reasoning solver."},
            {"role": "user", "content": prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        output = self.model.generate(
    **inputs,
    max_new_tokens=512,   # match config
    temperature=0.1,
    top_p=0.9,
    repetition_penalty=1.2,
    do_sample=True
)


        input_length = inputs["input_ids"].shape[1]
        new_tokens = output[0][input_length:]
        decoded = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        for char in decoded:
            if char in ["A", "B", "C", "D"]:
                return char

        return "A"
