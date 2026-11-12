import base64


EXAMPLE_IMAGE_PATH = "input/figure_4_1.png"
EXAMPLE_ANSWER_PATH = "ground_truth/figure_4_1_visible_ground_truth.txt"


# this function loads an image and encodes it in base64 format
def load_image(path):
    with open(path, "rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")


# This function an image message for the model
def image_message(path):
    image = load_image(path)

    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/png;base64,{image}"
        },
    }


# This function creates a one-shot prompt with an example diagram, which it then uses for the target diagram
def create_one_shot_messages(prompt, target_image):
    with open(EXAMPLE_ANSWER_PATH, encoding="utf-8") as file:
        example_answer = file.read()

    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Here is an example system diagram:",
                },
                image_message(EXAMPLE_IMAGE_PATH),
            ],
        },
        {
            "role": "assistant",
            "content": example_answer,
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                image_message(target_image),
            ],
        },
    ]
