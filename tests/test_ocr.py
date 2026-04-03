from tests.conftest import reload_module


def test_run_ocr_converts_polygons_to_bounding_boxes(monkeypatch):
    ocr_tool = reload_module("app.tools.ocr_tool")
    ocr_tool._reader = None

    class DummyReader:
        def readtext(self, image_path, *args, **kwargs):
            return [
                (
                    [(10.2, 20.9), (31.4, 20.1), (30.7, 44.8), (9.5, 45.2)],
                    "L-101",
                    0.97,
                ),
                (
                    [(50, 60), (80, 60), (80, 90), (50, 90)],
                    "PUMP-1",
                    0.88,
                ),
            ]

    monkeypatch.setattr(ocr_tool, "get_reader", lambda: DummyReader())
    monkeypatch.setattr(
        ocr_tool,
        "_prepare_variants",
        lambda image: [{"name": "gray", "image": image, "rotation": 0}],
    )

    results = ocr_tool.run_ocr("dummy.png")

    assert len(results) == 2
    assert results[0].text == "L-101"
    assert results[0].bbox == (9, 20, 31, 45)
    assert results[1].bbox == (50, 60, 80, 90)
    assert results[1].score == 0.88
