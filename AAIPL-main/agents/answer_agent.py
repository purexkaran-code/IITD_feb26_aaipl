import json
import argparse
from agents.answer_model import AnswerModel

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default="outputs/filtered_questions.json")
    parser.add_argument("--output_file", default="outputs/answers.json")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print("Opening generated questions...")
    with open(args.input_file) as f:
        questions = json.load(f)

    model = AnswerModel()
    answers = []

    for i, q in enumerate(questions):

        if args.verbose:
            print(f"Answering question {i+1} of {len(questions)}")

        ans = model.answer_question(q)
        answers.append(ans)

    with open(args.output_file, "w") as f:
        json.dump(answers, f, indent=2)

    print("✅ Answers saved:", args.output_file)

if __name__ == "__main__":
    main()
