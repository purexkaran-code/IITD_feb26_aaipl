import json
import argparse
from agents.question_model import QuestionModel

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", default="outputs/question__train.json")
    parser.add_argument("--num_questions", type=int, default=10)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    with open("assets/topics.json") as f:
        topics = json.load(f)
        
    flat_topics = []
    for category in topics.values():
        flat_topics.extend(category)

    model = QuestionModel()
    questions = []

    for i in range(args.num_questions):
        topic = topics[i % len(topics)]

        if args.verbose:
            print(f"Generating question {i+1} on {topic}")

        q = model.generate_question(topic)
        questions.append(q)

    with open(args.output_file, "w") as f:
        json.dump(questions, f, indent=2)

    print("✅ Questions saved:", args.output_file)

if __name__ == "__main__":
    main()
