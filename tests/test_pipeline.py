import json
from pathlib import Path

from app.schemas import OCRText, PipeSegment
from tests.conftest import reload_module


def test_mock_pipeline_end_to_end(monkeypatch, tmp_path):
    nodes = reload_module("app.graph.nodes")

    monkeypatch.setattr(
        nodes,
        "run_ocr",
        lambda image_path: [
            OCRText(text="200-P-310226-NB01-PP", bbox=(10, 10, 20, 20), score=0.99),
            OCRText(text="E-3118", bbox=(0, 0, 8, 8), score=0.91),
            OCRText(text="EA-3114", bbox=(100, 0, 120, 8), score=0.90),
        ],
    )
    monkeypatch.setattr(
        nodes,
        "get_image_quality_summary",
        lambda image_path: {
            "input_quality": {"label": "input", "width": 100, "height": 100, "sharpness": 10.0},
            "output_quality": {"label": "enhanced_output", "width": 400, "height": 400, "sharpness": 20.0},
        },
    )
    monkeypatch.setattr(nodes, "find_equipment_candidates", lambda texts: [])
    monkeypatch.setattr(
        nodes,
        "extract_pipe_segments",
        lambda image_path: [
            PipeSegment(id="seg_1", points=[(10, 10)], bbox=(12, 12, 24, 24)),
            PipeSegment(id="seg_2", points=[(25, 12)], bbox=(25, 12, 35, 24)),
        ],
    )
    monkeypatch.setattr(
        nodes,
        "select_target_pipe_by_text",
        lambda target_text, ocr_texts, pipe_segments: {
            "success": True,
            "anchor_text": "200-P-310226-NB01-PP",
            "anchor_bbox": (10, 10, 20, 20),
            "segment_id": "seg_1",
            "segment_bbox": (12, 12, 24, 24),
        },
    )
    monkeypatch.setattr(nodes, "build_pipe_graph", lambda segments: {"segments": segments})
    monkeypatch.setattr(
        nodes,
        "trace_connected_component",
        lambda graph, segment_id: {"segment_ids": ["seg_1", "seg_2"], "endpoints": [(0, 0), (100, 0)]},
    )

    def fake_save_overlay(image_path, all_segments, target_segment_ids, out_path):
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(b"overlay")
        return str(out_path)

    monkeypatch.setattr(nodes, "save_overlay", fake_save_overlay)
    monkeypatch.setattr(nodes, "OUTPUT_DIR", str(tmp_path))

    class DummyVLMTool:
        def infer_from_to(
            self,
            original_image_path,
            overlay_image_path,
            target_desc,
            path_hints=None,
            equipment_hints=None,
        ):
            assert Path(overlay_image_path).exists()
            return {
                "FROM": "E-3118",
                "TO": "EA-3114",
                "answer": "제공해 주신 P&ID를 확인한 결과, 시작은 E-3118이고 끝은 EA-3114입니다.",
                "confidence": 0.82,
                "reason": "mock pipeline",
                "evidence": {"notes": target_desc},
            }

    monkeypatch.setattr(nodes, "VLMTool", DummyVLMTool)

    state = {
        "image_path": "dummy.png",
        "target_text": "200-P-310226-NB01-PP",
        "target_desc": "이 이미지의 배관 200-P-310226-NB01-PP 의 시작과 끝 설비를 알려줘",
        "logs": [],
    }

    for fn in [
        nodes.ocr_node,
        nodes.equipment_node,
        nodes.pipe_node,
        nodes.target_pipe_node,
        nodes.path_trace_node,
        nodes.overlay_node,
        nodes.vlm_node,
    ]:
        state.update(fn(state))

    assert state["final_result"]["FROM"] == "E-3118"
    assert state["final_result"]["TO"] == "EA-3114"
    assert "E-3118" in state["final_result"]["answer"]
    assert Path(state["overlay_path"]).exists()


def test_vlm_is_skipped_when_target_pipe_not_found():
    nodes = reload_module("app.graph.nodes")

    state = {
        "image_path": "dummy.png",
        "target_text": "200-P-310225-NB01-HC",
        "target_desc": "이 이미지의 배관 200-P-310225-NB01-HC 의 시작과 끝 설비를 알려줘",
        "selected_pipe_seed": {"success": False, "reason": "not found"},
        "logs": [],
    }

    result = nodes.vlm_node(state)

    assert result["final_result"]["FROM"] == ""
    assert result["final_result"]["TO"] == ""
    assert "신뢰성 있게 판단할 수 없습니다" in result["final_result"]["answer"]
    assert "VLM skipped" in result["logs"][-1]


def test_main_writes_result_json_with_explicit_image_filename(monkeypatch, tmp_path, capsys):
    main_mod = reload_module("app.main")

    class DummyApp:
        def invoke(self, init_state):
            assert init_state["image_path"].replace("\\", "/").endswith("data/input/sample_pid.png")
            assert init_state["target_text"] == "200-P-310226-NB01-PP"
            assert "이 이미지의 배관" in init_state["target_desc"]
            return {
                "logs": ["step-a", "step-b"],
                "final_result": {
                    "FROM": "E-3118",
                    "TO": "EA-3114",
                    "answer": "제공해 주신 P&ID를 확인한 결과, 시작은 E-3118이고 끝은 EA-3114입니다.",
                    "confidence": 0.9,
                    "reason": "mocked",
                    "evidence": {"notes": "ok"},
                },
            }

    monkeypatch.setattr(main_mod, "build_workflow", lambda: DummyApp())
    monkeypatch.setattr(main_mod, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(
        main_mod,
        "sys",
        type(
            "DummySys",
            (),
            {
                "argv": [
                    "app.main",
                    "sample_pid.png",
                    "이 이미지의 배관 200-P-310226-NB01-PP 의 시작과 끝 설비를 알려줘",
                ]
            },
        )(),
    )

    main_mod.main()

    captured = capsys.readouterr()
    result_path = tmp_path / "result.json"

    assert result_path.exists()
    assert "FINAL RESULT" in captured.out
    assert "E-3118" in captured.out
    assert json.loads(result_path.read_text(encoding="utf-8"))["FROM"] == "E-3118"


def test_extract_target_text_from_question():
    main_mod = reload_module("app.main")
    question = "이 이미지의 배관 200-P-310226-NB01-PP 의 시작과 끝 설비를 알려줘"

    assert main_mod.extract_target_text(question) == "200-P-310226-NB01-PP"
