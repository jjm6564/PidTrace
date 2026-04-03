import base64
from typing import Any, Dict, List, Optional

from app.config import MODEL_PROVIDER, OLLAMA_BASE_URL, OLLAMA_MODEL
from app.prompt.vlm_prompts import VLM_SYSTEM_PROMPT, build_user_prompt
from app.schemas import FromToResult


class VLMTool:
    def __init__(self, model_name: Optional[str] = None):
        provider = MODEL_PROVIDER.lower()
        if provider == "ollama":
            from langchain_ollama import ChatOllama

            self.llm = ChatOllama(
                model=model_name or OLLAMA_MODEL,
                base_url=OLLAMA_BASE_URL,
                temperature=0,
            )
        elif provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            self.llm = ChatAnthropic(model=model_name or "claude-3-5-sonnet-latest", temperature=0)
        else:
            from langchain_openai import ChatOpenAI

            self.llm = ChatOpenAI(model=model_name or "gpt-4.1", temperature=0)

    def _to_data_url(self, image_path: str) -> str:
        with open(image_path, "rb") as file:
            b64 = base64.b64encode(file.read()).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    def infer_from_to(
        self,
        original_image_path: str,
        overlay_image_path: str,
        target_desc: str,
        path_hints: Optional[Dict[str, Any]] = None,
        equipment_hints: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        original_url = self._to_data_url(original_image_path)
        overlay_url = self._to_data_url(overlay_image_path)

        path_hint_text = "- no path hints"
        if path_hints:
            path_hint_text = (
                f"- selected segment count: {len(path_hints.get('segment_ids', []))}\n"
                f"- inferred path endpoints: {path_hints.get('endpoints', [])}\n"
                f"- target anchor text: {path_hints.get('anchor_text', '')}"
            )

        equipment_hint_text = "- no equipment hints"
        if equipment_hints:
            equipment_hint_text = "\n".join(f"- {item}" for item in equipment_hints[:10])

        structured_llm = self.llm.with_structured_output(FromToResult)

        messages = [
            ("system", VLM_SYSTEM_PROMPT),
            (
                "human",
                [
                    {
                        "type": "text",
                        "text": build_user_prompt(
                            target_desc=target_desc,
                            path_hint_text=path_hint_text,
                            equipment_hint_text=equipment_hint_text,
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": original_url}},
                    {"type": "image_url", "image_url": {"url": overlay_url}},
                ],
            ),
        ]

        result = structured_llm.invoke(messages)
        return result.model_dump()
