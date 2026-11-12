import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "from papers"))
from ariadne import run_ariadne

import json
import base64
import os
from datetime import datetime
from methods.voting import vote_results

from dotenv import load_dotenv
from openai import OpenAI

from methods.ocr import run_ocr, add_ocr
from methods.one_shot import create_one_shot_messages

from config import (MODEL, IMAGE_PATH, USE_TWO_STAGE, USE_CONTEXT, USE_OCR,
                    USE_NOTATION, USE_ONE_SHOT, USE_LITERATURE_BASELINE, USE_ARIADNE)

NUMBER_OF_RUNS = 5

NOTATION_FILES = {
    "figure_1": "resources/notation/notation_figure_1.txt",
    "figure_4": "resources/notation/notation_figure_4.txt",
    "figure_7": "resources/notation/notation_figure_7.txt",
    "figure_13_1": "resources/notation/notation_figure_13_1.txt",
    "figure_A_3": "resources/notation/notation_figure_A_3.txt",
    "tank": "resources/notation/notation_tank.txt",
}

# This function reads text from a file
def load_text(path):
    with open(path, encoding="utf-8") as file:
        return file.read()


# This function finds the relevant notation file for a system diagram
def get_notation_path(image_path):
    figure_name = Path(image_path).stem

    return NOTATION_FILES[figure_name]


# This function loads an image and encodes it in base64 format
def load_image(path):
    with open(path, "rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")


# This function creates a message for the model that includes the prompt and the image
def create_messages(prompt, image_path):
    content = [
        {"type": "text", "text": prompt},
        {"type": "text", "text": "Original system diagram:"},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{load_image(image_path)}"
            },
        },
    ]

    return [
        {
            "role": "user",
            "content": content,
        }
    ]


# This function sends a prompt and diagram to the selected model
def ask_model(
    client,
    prompt,
    image_path,
):
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=create_messages(
            prompt,
            image_path,
        ),
    )

    return response.choices[0].message.content


# This function adds context, notation or OCR when they are enabled in the config file
def add_extra_information(prompt, image_path, ocr_text=None):
    if USE_CONTEXT:
        context = load_text("resources/context_rules.txt")
        prompt = f"{context}\n\n{prompt}"

    if USE_NOTATION:
        notation_path = get_notation_path(image_path)
        notation = load_text(notation_path)
        prompt = f"{notation}\n\n{prompt}"

    if USE_OCR:
        if ocr_text is None:
            ocr_text = run_ocr(image_path)

        prompt = add_ocr(prompt, ocr_text)

    return prompt


# This function identifies components first, then connections
def run_two_stage(client, image_path):
    ocr_text = None

    if USE_OCR:
        ocr_text = run_ocr(image_path)

    components_prompt = load_text(
        "prompts/two_stage_components.txt"
    )

    components_prompt = add_extra_information(
        components_prompt,
        image_path,
        ocr_text,
    )

    components = ask_model(
        client,
        components_prompt,
        image_path,
    )

    connections_prompt = load_text(
        "prompts/two_stage_connections.txt"
    )

    connections_prompt = connections_prompt.replace(
        "{components}",
        components,
    )

    connections_prompt = add_extra_information(
        connections_prompt,
        image_path,
        ocr_text,
    )

    connections = ask_model(
        client,
        connections_prompt,
        image_path,
    )

    result = f"{components}\n\n{connections}"

    prompt = (
        f"{components_prompt}\n\n"
        f"{connections_prompt}"
    )

    return result, prompt


# This function converts the result from the literature baseline into the same format as the other methods
def convert_literature_baseline_result(result):
    text = result.strip()

    if text.startswith("```"):
        parts = text.split("```", 2)

        if len(parts) < 3:
            raise ValueError("Unclosed JSON code block")

        text = parts[1].strip()

        if text.lower().startswith("json\n"):
            text = text.split("\n", 1)[1].strip()

    data = json.loads(text)

    components = [
        node["id"] for node in data["nodes"]
    ]

    for edge in data["edges"]:
        if type(edge.get("directed")) is not bool:
            raise ValueError("Invalid directed value")

    connections = [
        f'{edge["source"]} {"->" if edge["directed"] else "||"} {edge["target"]}'
        for edge in data["edges"]
    ]

    return (
        "Components:\n"
        + "\n".join(f"- {c}" for c in components)
        + "\n\nConnections:\n"
        + "\n".join(f"- {c}" for c in connections)
    )


# The function saves the model output and other relevant information to a text file in the results folder
def save_result(result, image_path, prompt, method, raw_result=None):
    os.makedirs("results", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = f"results/result_{timestamp}.txt"

    with open(path, "w", encoding="utf-8") as file:
        file.write(f"Model: {MODEL}\n")
        file.write(f"Method: {method}\n")
        file.write(f"Image: {IMAGE_PATH}\n")
        file.write(f"Processed image: {image_path}\n")
        file.write(f"Context: {USE_CONTEXT}\n")
        file.write(f"OCR: {USE_OCR}\n")
        file.write(f"Notation: {USE_NOTATION}\n")
        file.write(f"One-shot: {USE_ONE_SHOT}\n")
        file.write(f"Two-stage: {USE_TWO_STAGE}\n")
        file.write("\n--- PROMPT ---\n\n")
        file.write(prompt)
        if raw_result is not None:
            file.write("\n--- RAW JSON ---\n\n")
            file.write(raw_result)

        file.write("\n\n--- MODEL OUTPUT ---\n\n")
        file.write(result)

    print(f"\nSaved to {path}")


# This function runs the selected experiment five times and then uses majority voting
def main():
    load_dotenv()

    client = OpenAI(
        base_url="https://llm.hpc.ntnu.no/v1",
        api_key=os.getenv("NTNU_API_KEY"),
    )

    image_path = IMAGE_PATH
    results = []

    for run_number in range(1, NUMBER_OF_RUNS + 1):
        print(f"\n\nRun {run_number}/{NUMBER_OF_RUNS}:", end="")

        if USE_TWO_STAGE and USE_ONE_SHOT:
            raise ValueError(
                "One-shot is not currently supported with two-stage"
            )


        if USE_ARIADNE:
            method = "Ariadne-inspired"

            result, prompt = run_ariadne(
                client, image_path, ask_model, load_text
            )

        elif USE_TWO_STAGE:
            method = "two-stage"

            result, prompt = run_two_stage(
                client,
                image_path,
            )

        else:
            if USE_LITERATURE_BASELINE:
                method = "Literature Baseline"
                prompt = load_text("from papers/literature_baseline.txt")

            elif USE_ONE_SHOT:
                method = "one-shot"
                prompt = load_text("prompts/one_shot_image.txt")

            else:
                method = "zero-shot"
                prompt = load_text("prompts/zero_shot_image.txt")

            if USE_CONTEXT:
                context = load_text("resources/context_rules.txt")
                prompt = f"{context}\n\n{prompt}"

            if USE_OCR:
                ocr_text = run_ocr(image_path)
                prompt = add_ocr(prompt, ocr_text)

            if USE_NOTATION:
                notation_path = get_notation_path(image_path)
                notation = load_text(notation_path)
                prompt = f"{notation}\n\n{prompt}"

            if USE_ONE_SHOT:
                messages = create_one_shot_messages(
                    prompt,
                    image_path,
                )

            else:
                messages = create_messages(
                    prompt,
                    image_path,
                )

            response = client.chat.completions.create(
                model=MODEL,
                temperature=0,
                messages=messages,
            )

            raw_result = response.choices[0].message.content

            if USE_LITERATURE_BASELINE:
                try:
                    result = convert_literature_baseline_result(raw_result)
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                    print(f"\nJSON format error: {e}")
                    result = "Components:\n\nConnections:\n"
            else:
                result = raw_result

        print(result)

        save_result(
            result,
            image_path,
            prompt,
            f"{method} - run {run_number}",
            raw_result=raw_result if method == "Literature Baseline" else None,

        )

        results.append(result)

    voted_result = vote_results(results)

    print("\n\nVoted result from 5 runs:\n")
    print(voted_result)

    save_result(
        voted_result,
        image_path,
        prompt,
        f"{method} - voting",
    )


if __name__ == "__main__":
    main()
