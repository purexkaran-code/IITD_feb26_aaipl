import json
import torch
from unsloth import FastLanguageModel

class AnswerModel:
    def __init__(self):

        base_model = "Qwen/Qwen2.5-14B-Instruct"

        print("Loading Answer Model...")

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model,
            dtype=torch.bfloat16,
            load_in_4bit=False,
        )

        # Load LoRA if exists
        try:
            self.model.load_adapter("a_agent_lora")
            print("✅ LoRA adapter loaded")
        except:
            print("⚠️ No LoRA found — using base model")

        self.model.eval()

    def answer_question(self, question_data):

        question_text = question_data["question"]
        choices = "\n".join(question_data["choices"])

        prompt = f"""
Solve the logical reasoning question.

{question_text}

Choices:
{choices}

Respond ONLY in JSON:

{{"answer":"A","reasoning":"brief explanation"}}
"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        output = self.model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.1,
            top_p=0.9
        )

        text = self.tokenizer.decode(output[0], skip_special_tokens=True)

        start = text.find("{")
        end = text.rfind("}") + 1

        return json.loads(text[start:end])
