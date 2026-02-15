import json
from agents.question_model import QuestionModel


def main():
    with open("assets/topics.json") as f:
        topic_data = json.load(f)

    import random
    category = random.choice(list(topic_data.keys()))
    subtopic = random.choice(topic_data[category])
    
    topic = subtopic


    model = QuestionModel()
    question = model.generate_question(topic)

    print(json.dumps(question))


if __name__ == "__main__":
    main()
