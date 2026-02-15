import json
from agents.answer_model import AnswerModel

def main():
    import sys
    question_json = json.load(sys.stdin)

    model = AnswerModel()
    answer = model.answer_question(question_json)

    print(json.dumps({"answer": answer}))

if __name__ == "__main__":
    main()
