"""Headless Plotly/Jupyter views over cached observations and replayed memory."""

from dataclasses import replace
from functools import partial
from html import escape
from io import BytesIO

import numpy as np
import plotly.graph_objects as go
import pycocotools.mask as mask_utils
from PIL import Image, ImageDraw, ImageFont

from graph.association import valid_box
from graph.edges import build_edges
from graph.node import Node
from graph.pipeline import cached_observations, finish, replay, save_result
from graph.query import rank
from utils import footage


def color(object_id):
    return np.random.default_rng(int(object_id)).integers(65, 230, size=3)


def memory_nodes(memory) -> list[Node]:
    """The memory's objects as graph.node.Node, for graph.edges. Objects without a box get a NaN box."""
    nodes = []
    for n in sorted(memory.nodes.values(), key=lambda n: n.id):
        lo, hi = (n.box.min, n.box.max) if valid_box(n.box) else (np.full(3, np.nan), np.full(3, np.nan))
        nodes.append(Node(n.id, n.label or "", lo, hi, np.empty(0, int), np.empty(0, np.int32), []))
    return nodes


def boxes_figure(memory, selected=None):
    fig = go.Figure()
    corners_bits = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)])
    pairs = [(i, j) for i in range(8) for j in range(i + 1, 8) if np.sum(corners_bits[i] != corners_bits[j]) == 1]
    for node in memory.nodes.values():
        if not valid_box(node.box):
            continue
        corners = np.where(corners_bits, node.box.max, node.box.min)
        xyz = np.array([p for i, j in pairs for p in (corners[i], corners[j], [np.nan] * 3)])
        rgb = color(node.id)
        fig.add_trace(
            go.Scatter3d(
                x=xyz[:, 0],
                y=xyz[:, 1],
                z=xyz[:, 2],
                mode="lines",
                line={"color": f"rgb({rgb[0]},{rgb[1]},{rgb[2]})", "width": 7 if selected == node.id else 3},
                name=f"{node.id}: {node.label}",
                showlegend=False,
            )
        )
        center = node.box.center
        fig.add_trace(
            go.Scatter3d(
                x=[center[0]],
                y=[center[1]],
                z=[center[2]],
                mode="text",
                text=[f"{node.id}: {node.label}"],
                showlegend=False,
            )
        )
    for edge in build_edges(memory_nodes(memory)):
        a, b = (memory.nodes[key].box.center for key in (edge.a, edge.b))
        fig.add_trace(
            go.Scatter3d(
                x=[a[0], b[0]],
                y=[a[1], b[1]],
                z=[a[2], b[2]],
                mode="lines",
                line={"color": "gray" if edge.relation == "near" else "orange", "width": 2},
                text=f"{edge.relation}: {edge.gap:.2f} m",
                hoverinfo="text",
                showlegend=False,
            )
        )
    fig.update_layout(
        height=650,
        title="Persistent object memory (includes currently unseen objects)",
        scene={"aspectmode": "data", "xaxis_title": "x (m)", "yaxis_title": "y (m)", "zaxis_title": "z (m)"},
    )
    return fig


def overlay_rgb(run, masks, ids, frame, selected=None):
    """RGB mask overlay and ID label positions, independent of Plotly rendering."""
    image = footage.load_rgb(run, int(frame)).copy()
    annotations = []
    rows = np.flatnonzero(masks.frame == frame)
    for node in np.unique(ids[rows]):
        object_id = int(node)
        if object_id < 0 or (selected is not None and selected != object_id):
            continue
        mask = mask_utils.decode(mask_utils.merge([masks.rle[r] for r in rows[ids[rows] == node]])).astype(bool)
        image[mask] = (0.55 * image[mask] + 0.45 * color(object_id)).astype(np.uint8)
        v, u = np.nonzero(mask)
        if len(u):
            annotations.append(
                {
                    "x": float(np.median(u)),
                    "y": float(np.median(v)),
                    "text": str(object_id),
                    "showarrow": False,
                    "font": {"color": "white", "size": 16},
                    "bgcolor": "black",
                }
            )
    return image, annotations


def overlay_png(run, masks, ids, frame, selected=None):
    """Standard PNG bytes for Jupyter and native image widgets, with ID labels."""
    pixels, annotations = overlay_rgb(run, masks, ids, frame, selected)
    image = Image.fromarray(pixels)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=18)
    for label in annotations:
        position = (label["x"], label["y"])
        bounds = draw.textbbox(position, label["text"], font=font)
        draw.rectangle(bounds, fill="black")
        draw.text(position, label["text"], font=font, fill="white")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def overlay_figure(run, masks, ids, frame, selected=None):
    image, annotations = overlay_rgb(run, masks, ids, frame, selected)
    fig = go.Figure(go.Image(z=image))
    fig.update_layout(
        title=f"RGB + SAM masks, persistent IDs — frame {frame}",
        annotations=annotations,
        height=500,
        margin={"l": 10, "r": 10, "t": 45, "b": 10},
    )
    return fig


def scores_figure(matches):
    names = [
        "Frame",
        "Object",
        "Candidate",
        "SAM IDs",
        "Decision",
        "DINO",
        "Inside",
        "Center",
        "Geometry",
        "Score",
    ]
    values = [[] for _ in names]
    for match in matches:
        scores = match["scores"] or {}
        row = [
            match["frame"],
            match["object_id"],
            match["candidate_id"],
            str(match["sam_ids"]),
            "matched" if match["matched"] else "new",
        ]
        row += [
            "—" if key not in scores else f"{scores[key]:.3f}"
            for key in ("dino", "inside", "center", "geometry", "score")
        ]
        for column, value in zip(values, row, strict=True):
            column.append(value)
    fig = go.Figure(go.Table(header={"values": names}, cells={"values": values}))
    fig.update_layout(
        title="Pre-update selected/best compatible candidate scores", height=min(800, 100 + 28 * len(matches))
    )
    return fig


class GraphExplorer:
    """Loads no models. Retains only one prefix replay for the selected frame."""

    def __init__(self, run, config):
        self.run = run
        self.data = cached_observations(run)
        self.frames = footage.frame_numbers(run)
        self._text_cache = {}
        self.rebuild(config)

    def rebuild(self, config):
        result = replay(*self.data, config=config, frames=self.frames)
        self.config = config
        self.result = result
        self._prefix = None
        return self.result

    def at(self, frame):
        if self._prefix is None or self._prefix[0] != frame:
            self._prefix = (frame, replay(*self.data, config=self.config, frames=self.frames, until=frame))
        return self._prefix[1]

    def save(self):
        save_result(self.run.stage_dir("graph"), self.result, partial(finish, self.run, *self.data))

    def search(self, text, top=5, encoder=None):
        """Rank remembered objects by SigLIP text similarity, without image inference."""
        text = text.strip().lower()
        if not text:
            raise ValueError("Enter an object description, such as chair")
        if top < 1:
            raise ValueError("top must be positive")
        nodes = [n for n in self.result.memory.nodes.values() if n.siglip is not None]
        if not nodes:
            return []
        objects, embeddings = [n.id for n in nodes], np.stack([n.siglip for n in nodes])
        if text not in self._text_cache:
            if encoder is None:
                from models.siglip import load_encoder

                encoder = load_encoder()
            self._text_cache[text] = encoder.encode_text([text]).cpu().numpy()
        order, scores = rank(self._text_cache[text], embeddings)
        return [(int(objects[row]), float(score)) for row, score in zip(order[0, :top], scores[0, :top], strict=True)]

    def widgets(self):
        import ipywidgets as widgets
        from IPython.display import display

        output = widgets.Output()
        frame_image = widgets.Image(format="png", layout=widgets.Layout(width="100%", max_width="1100px"))
        frame_caption = widgets.HTML()
        observation_images = widgets.VBox()
        status = widgets.HTML()
        frame = widgets.SelectionSlider(
            options=[int(f) for f in self.frames],
            description="Frame",
            continuous_update=False,
            layout=widgets.Layout(width="80%"),
        )
        objects = widgets.Dropdown(description="Object", options=[("All objects", -1)])
        search_text = widgets.Text(placeholder="chair, striped pillow, wooden table…", description="Search")
        search_button = widgets.Button(description="Find objects", button_style="info")
        search_status = widgets.HTML(value="Search all remembered objects using SigLIP text similarity.")
        search_results = widgets.Dropdown(
            description="Matches",
            options=[("Search to see matches", -1)],
            disabled=True,
            layout=widgets.Layout(width="80%"),
        )
        c = self.config

        def slider(name, value, low=0, high=1, step=0.01):
            return widgets.FloatSlider(
                description=name, value=value, min=low, max=high, step=step, continuous_update=False
            )

        controls = {
            "dino_weight": slider("DINO weight", c.dino_weight, high=2),
            "box_weight": slider("3D weight", c.box_weight, high=2),
            "threshold": slider("Threshold", c.threshold),
            "max_descriptors": widgets.IntSlider(description="Bank size", value=c.max_descriptors, min=1, max=16),
            "max_candidate_distance": slider("Gate (m)", c.max_candidate_distance or 1.5, low=0.05, high=5, step=0.05),
            "geometry_inside_weight": slider("Inside share", c.geometry_inside_weight),
            "center_distance_scale": slider("Center (m)", c.center_distance_scale, low=0.05, high=3, step=0.05),
            "enable_dino": widgets.Checkbox(description="Use DINO", value=c.enable_dino),
            "enable_geometry": widgets.Checkbox(description="Use geometry", value=c.enable_geometry),
            "require_same_label": widgets.Checkbox(description="Same label", value=c.require_same_label),
        }
        rebuild_button = widgets.Button(description="Rebuild from cache", button_style="primary")
        save_button = widgets.Button(description="Save graph + config")

        def refresh_objects():
            objects.options = [("All objects", -1)] + [
                (f"{n.id}: {n.label}", n.id) for n in self.result.memory.nodes.values()
            ]

        def render(change=None):
            try:
                selected = None if objects.value == -1 else objects.value
                prefix = self.at(frame.value)
                frame_image.value = overlay_png(self.run, self.data[0], prefix.graph.node, frame.value, selected)
                frame_caption.value = f"<b>RGB + SAM masks, persistent IDs — frame {frame.value}</b>"
                observation_images.children = ()
                output.clear_output(wait=True)
                # Explicit output updates work from widget callbacks without
                # depending on capture of the currently executing cell.
                output.append_display_data(boxes_figure(prefix.memory, selected))
                history = self.result.history
                output.append_display_data(
                    go.Figure(
                        go.Scatter(x=[h["frame"] for h in history], y=[h["nodes"] for h in history], mode="lines")
                    ).update_layout(
                        title="Persistent nodes over time", xaxis_title="Frame", yaxis_title="Nodes", height=300
                    )
                )
                matches = [
                    m
                    for m in self.result.matches
                    if (m["frame"] == frame.value if selected is None else m["object_id"] == selected)
                ]
                output.append_display_data(scores_figure(matches))
                if selected is not None:
                    observed = sorted({m["frame"] for m in matches})
                    panels = [widgets.HTML(value=f"<b>Object {selected}: {len(observed)} observed frames</b>")]
                    # Show representative frames across the whole sequence, including revisits.
                    for index in np.unique(
                        np.linspace(0, max(len(observed) - 1, 0), min(6, len(observed))).astype(int)
                    ):
                        number = observed[index]
                        panels.extend(
                            [
                                widgets.HTML(value=f"Frame {number}"),
                                widgets.Image(
                                    value=overlay_png(self.run, self.data[0], self.result.graph.node, number, selected),
                                    format="png",
                                    layout=widgets.Layout(width="100%", max_width="1000px"),
                                ),
                            ]
                        )
                    observation_images.children = tuple(panels)
            except Exception as exc:
                status.value = f"Visualization failed: {exc}"
                raise

        def rebuild_clicked(button):
            try:
                config = replace(self.config, **{key: control.value for key, control in controls.items()})
                status.value = "Rebuilding cached observations…"
                self.rebuild(config)
                refresh_objects()
                search_results.options = [("Search again after rebuilding", -1)]
                search_results.disabled = True
                status.value = f"{len(self.result.memory.nodes)} nodes; normalized weights {config.effective_weights}"
                render()
            except Exception as exc:
                status.value = f"Rebuild failed: {exc}"

        def save_clicked(button):
            self.save()
            path = self.run.stage_dir("graph") / "association_config.json"
            import json
            from dataclasses import asdict

            path.write_text(json.dumps(asdict(self.config), indent=2))
            status.value = f"Saved graph and reusable config to {path}"

        def select_search_result(change):
            object_id = search_results.value
            if object_id is None or object_id < 0:
                return
            observed = self.data[0].frame[self.result.graph.node == object_id]
            if len(observed) and frame.value not in observed:
                frame.value = int(observed[np.abs(observed - frame.value).argmin()])
            objects.value = object_id

        def search_clicked(button):
            search_button.disabled = True
            try:
                search_status.value = "Searching… First search loads the SigLIP encoder on the GPU."
                matches = self.search(search_text.value)
                search_results.options = [("Choose a match", -1)] + [
                    (f"{score:.3f} — object {object_id}: {self.result.memory.nodes[object_id].label}", object_id)
                    for object_id, score in matches
                ]
                search_results.disabled = not bool(matches)
                search_status.value = (
                    f"Best matches for <b>{escape(search_text.value.strip())}</b>; scores are cosine similarity."
                    if matches
                    else "No objects have valid SigLIP descriptors."
                )
                if matches:
                    search_results.value = matches[0][0]
            except Exception as exc:
                search_status.value = f"Search failed: {escape(str(exc))}"
            finally:
                search_button.disabled = False

        refresh_objects()
        frame.observe(render, names="value")
        objects.observe(render, names="value")
        rebuild_button.on_click(rebuild_clicked)
        save_button.on_click(save_clicked)
        search_button.on_click(search_clicked)
        search_results.observe(select_search_result, names="value")
        self.ui = {
            "controls": controls,
            "frame": frame,
            "objects": objects,
            "rebuild": rebuild_button,
            "save": save_button,
            "status": status,
            "frame_image": frame_image,
            "observation_images": observation_images,
            "output": output,
            "search_text": search_text,
            "search": search_button,
            "search_results": search_results,
            "search_status": search_status,
        }
        display(
            widgets.VBox(
                [
                    widgets.HBox(
                        [widgets.VBox(list(controls.values())[:6]), widgets.VBox(list(controls.values())[6:])]
                    ),
                    widgets.HBox([rebuild_button, save_button]),
                    status,
                    frame,
                    objects,
                    widgets.HBox([search_text, search_button]),
                    search_status,
                    search_results,
                    frame_caption,
                    frame_image,
                    observation_images,
                    output,
                ]
            )
        )
        render()
