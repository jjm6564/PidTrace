import importlib
import sys
import types


def ensure_stub_modules():
    if "easyocr" not in sys.modules:
        easyocr = types.ModuleType("easyocr")

        class Reader:
            def __init__(self, *args, **kwargs):
                pass

            def readtext(self, image_path, *args, **kwargs):
                return []

        easyocr.Reader = Reader
        sys.modules["easyocr"] = easyocr

    if "cv2" not in sys.modules:
        cv2 = types.ModuleType("cv2")
        cv2.COLOR_BGR2GRAY = 0
        cv2.INTER_CUBIC = 0
        cv2.INTER_LANCZOS4 = 1
        cv2.THRESH_BINARY = 0
        cv2.THRESH_OTSU = 0
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C = 0
        cv2.THRESH_BINARY_INV = 0
        cv2.MORPH_OPEN = 0
        cv2.RETR_LIST = 0
        cv2.CHAIN_APPROX_SIMPLE = 0
        cv2.ROTATE_90_CLOCKWISE = 0
        cv2.ROTATE_90_COUNTERCLOCKWISE = 1

        def make_image(shape=(100, 100, 3)):
            return types.SimpleNamespace(shape=shape)

        cv2.imread = lambda path: make_image()
        cv2.cvtColor = lambda image, code: make_image((image.shape[0], image.shape[1]))
        cv2.resize = lambda image, dsize=None, fx=1.0, fy=1.0, interpolation=None: make_image(
            (max(1, int(image.shape[0] * fy)), max(1, int(image.shape[1] * fx)))
        )
        cv2.rotate = lambda image, flag: make_image((image.shape[1], image.shape[0]))
        cv2.fastNlMeansDenoising = lambda src, dst=None, h=3, templateWindowSize=7, searchWindowSize=21: src
        cv2.GaussianBlur = lambda image, ksize, sigmaX: image
        cv2.addWeighted = lambda src1, alpha, src2, beta, gamma: src1
        cv2.createCLAHE = lambda clipLimit=2.0, tileGridSize=(8, 8): types.SimpleNamespace(apply=lambda image: image)
        cv2.threshold = lambda gray, a, b, c: (None, gray)
        cv2.adaptiveThreshold = lambda src, maxValue, adaptiveMethod, thresholdType, blockSize, C: src
        cv2.morphologyEx = lambda binary, op, kernel: binary
        cv2.findContours = lambda clean, mode, method: ([], None)
        cv2.contourArea = lambda cnt: 0
        cv2.boundingRect = lambda cnt: (0, 0, 0, 0)
        cv2.rectangle = lambda *args, **kwargs: None
        cv2.imwrite = lambda *args, **kwargs: True
        sys.modules["cv2"] = cv2

    if "torch" not in sys.modules:
        torch = types.ModuleType("torch")
        torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        sys.modules["torch"] = torch

    if "networkx" not in sys.modules:
        networkx = types.ModuleType("networkx")

        class DegreeView:
            def __init__(self, adj):
                self.adj = adj

            def __getitem__(self, node_id):
                return len(self.adj.get(node_id, set()))

        class Graph:
            def __init__(self):
                self.nodes = {}
                self.adj = {}

            def add_node(self, node_id, **attrs):
                self.nodes[node_id] = attrs
                self.adj.setdefault(node_id, set())

            def add_edge(self, a, b, **attrs):
                self.adj.setdefault(a, set()).add(b)
                self.adj.setdefault(b, set()).add(a)

            def __contains__(self, item):
                return item in self.adj

            @property
            def degree(self):
                return DegreeView(self.adj)

            def subgraph(self, nodes):
                sub = Graph()
                node_set = set(nodes)
                for node_id in node_set:
                    sub.add_node(node_id, **self.nodes.get(node_id, {}))
                for node_id in node_set:
                    for neighbor in self.adj.get(node_id, set()):
                        if neighbor in node_set:
                            sub.add_edge(node_id, neighbor)
                return sub

        def node_connected_component(graph, segment_id):
            visited = set()
            stack = [segment_id]
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                stack.extend(graph.adj.get(node, set()) - visited)
            return visited

        networkx.Graph = Graph
        networkx.node_connected_component = node_connected_component
        sys.modules["networkx"] = networkx

    if "langgraph.graph" not in sys.modules:
        langgraph = sys.modules.setdefault("langgraph", types.ModuleType("langgraph"))
        graph_mod = types.ModuleType("langgraph.graph")
        graph_mod.START = "__start__"
        graph_mod.END = "__end__"

        class StateGraph:
            def __init__(self, state_type):
                self.nodes = {}
                self.edges = []

            def add_node(self, name, fn):
                self.nodes[name] = fn

            def add_edge(self, src, dst):
                self.edges.append((src, dst))

            def compile(self):
                graph = self

                class CompiledGraph:
                    def invoke(self, state):
                        current = dict(state)
                        order = [
                            "ocr",
                            "equipment",
                            "pipe",
                            "target_pipe",
                            "path_trace",
                            "overlay",
                            "vlm",
                        ]
                        for name in order:
                            if name in graph.nodes:
                                current.update(graph.nodes[name](current))
                        return current

                return CompiledGraph()

        graph_mod.StateGraph = StateGraph
        sys.modules["langgraph.graph"] = graph_mod
        setattr(langgraph, "graph", graph_mod)

    if "langchain_openai" not in sys.modules:
        lc_openai = types.ModuleType("langchain_openai")

        class ChatOpenAI:
            def __init__(self, *args, **kwargs):
                pass

            def with_structured_output(self, schema):
                class Structured:
                    def invoke(self, messages):
                        return schema(
                            FROM="MockFROM",
                            TO="MockTO",
                            answer="mocked",
                            confidence=0.5,
                            reason="mocked",
                            evidence={"notes": "mocked"},
                        )

                return Structured()

        lc_openai.ChatOpenAI = ChatOpenAI
        sys.modules["langchain_openai"] = lc_openai

    if "langchain_ollama" not in sys.modules:
        lc_ollama = types.ModuleType("langchain_ollama")

        class ChatOllama:
            def __init__(self, *args, **kwargs):
                pass

            def with_structured_output(self, schema):
                class Structured:
                    def invoke(self, messages):
                        return schema(
                            FROM="MockFROM",
                            TO="MockTO",
                            answer="mocked",
                            confidence=0.5,
                            reason="mocked",
                            evidence={"notes": "mocked"},
                        )

                return Structured()

        lc_ollama.ChatOllama = ChatOllama
        sys.modules["langchain_ollama"] = lc_ollama

    if "langchain_anthropic" not in sys.modules:
        lc_anthropic = types.ModuleType("langchain_anthropic")

        class ChatAnthropic:
            def __init__(self, *args, **kwargs):
                pass

            def with_structured_output(self, schema):
                class Structured:
                    def invoke(self, messages):
                        return schema(
                            FROM="MockFROM",
                            TO="MockTO",
                            answer="mocked",
                            confidence=0.5,
                            reason="mocked",
                            evidence={"notes": "mocked"},
                        )

                return Structured()

        lc_anthropic.ChatAnthropic = ChatAnthropic
        sys.modules["langchain_anthropic"] = lc_anthropic


ensure_stub_modules()


def reload_module(module_name: str):
    if module_name in sys.modules:
        return importlib.reload(sys.modules[module_name])
    return importlib.import_module(module_name)
