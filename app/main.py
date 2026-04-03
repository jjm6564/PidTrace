import json
import os
import re
import sys

from app.config import INPUT_IMAGE_PATH, OUTPUT_DIR
from app.graph.workflow import build_workflow


PIPE_NAME_PATTERN = re.compile(r"\b[A-Z0-9]{1,4}-P-\d{4,6}[A-Z0-9-]*\b|\b[A-Z]{1,4}-\d-\d{4}\b", re.IGNORECASE)


def extract_target_text(question: str) -> str:
    match = PIPE_NAME_PATTERN.search(question)
    if match:
        return match.group(0)
    return question.strip()


def resolve_image_path(image_arg: str) -> str:
    if os.path.isabs(image_arg):
        return image_arg

    if os.path.exists(image_arg):
        return image_arg

    candidate = os.path.join("data", "input", image_arg)
    return candidate


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if len(sys.argv) < 2:
        raise ValueError('Usage: python -m app.main [image_filename] "Ask a question about a pipe in the image"')

    if len(sys.argv) >= 3:
        image_path = resolve_image_path(sys.argv[1].strip())
        question = sys.argv[2].strip()
    else:
        image_path = INPUT_IMAGE_PATH
        question = sys.argv[1].strip()

    target_text = extract_target_text(question)

    app = build_workflow()

    init_state = {
        "image_path": image_path,
        "target_text": target_text,
        "target_desc": question,
        "logs": [],
    }

    result = app.invoke(init_state)
    final_result = result.get("final_result", {})
    final_answer = final_result.get("answer") or final_result.get("reason", "")

    print("\n[LOGS]")
    for log in result.get("logs", []):
        print("-", log)

    print("\n[FINAL RESULT]")
    if final_answer:
        print(final_answer)
    else:
        print(json.dumps(final_result, indent=2, ensure_ascii=False))

    with open(os.path.join(OUTPUT_DIR, "result.json"), "w", encoding="utf-8") as file:
        json.dump(final_result, file, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
