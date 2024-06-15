"""Utilities to visualize a graph."""

import numpy as np
import pygraphviz as pgv
import matplotlib
import matplotlib.cm

from sae_eap.graph.edge import Edge
from sae_eap.graph.graph import Graph


EDGE_TYPE_COLORS = {
    "q": "#FF00FF",  # Purple
    "k": "#00FF00",  # Green
    "v": "#0000FF",  # Blue
    "na": "#000000",  # Black
}


def cmap(cmap_name, rgb_order=False):
    """
    Extract colormap color information as a LUT compatible with cv2.applyColormap().
    Default channel order is BGR.

    Args:
        cmap_name: string, name of the colormap.
        rgb_order: boolean, if false or not set, the returned array will be in
                   BGR order (standard OpenCV format). If true, the order
                   will be RGB.

    Returns:
        A numpy array of type uint8 containing the colormap.
    """

    c_map = matplotlib.colormaps.get_cmap(cmap_name).resampled(256)
    rgba_data = matplotlib.cm.ScalarMappable(cmap=c_map).to_rgba(
        np.arange(0, 1.0, 1.0 / 256.0), bytes=True
    )
    rgba_data = rgba_data[:, 0:-1].reshape((256, 1, 3))

    # Convert to BGR (or RGB), uint8, for OpenCV.
    cmap = np.zeros((256, 1, 3), np.uint8)

    if not rgb_order:
        cmap[:, :, :] = rgba_data[:, :, ::-1]
    else:
        cmap[:, :, :] = rgba_data[:, :, :]

    return cmap


def color(cmap_name, index, rgb_order=False):
    """Returns a color of a given colormap as a list of 3 BGR or RGB values.

    Args:
        cmap_name: string, name of the colormap.
        index:     floating point between 0 and 1 or integer between 0 and 255,
                   index of the requested color.
        rgb_order: boolean, if false or not set, the returned list will be in
                   BGR order (standard OpenCV format). If true, the order
                   will be RGB.

    Returns:
        List of RGB or BGR values.
    """

    # Float values: scale from 0-1 to 0-255.
    if isinstance(index, float):
        val = round(min(max(index, 0.0), 1.0) * 255)
    else:
        val = min(max(index, 0), 255)

    # Get colormap and extract color.
    colormap = cmap(cmap_name, rgb_order)
    return colormap[int(val), 0, :].tolist()


def rgb2hex(rgb):
    """
    https://stackoverflow.com/questions/3380726/converting-an-rgb-color-tuple-to-a-hexidecimal-string
    """
    return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])


def generate_random_color(colorscheme: str) -> str:
    """
    https://stackoverflow.com/questions/28999287/generate-random-colors-rgb
    """

    return rgb2hex(color(colorscheme, np.random.randint(0, 256), rgb_order=True))


def get_color(edge: Edge):
    return EDGE_TYPE_COLORS[edge.type]


def to_graphviz(
    graph: Graph,
    colorscheme: str = "Pastel2",
    minimum_penwidth: float = 0.6,
    maximum_penwidth: float = 5.0,
    layout: str = "dot",
    seed: int | None = None,
) -> pgv.AGraph:
    """
    Colorscheme: a cmap colorscheme
    """
    g = pgv.AGraph(
        directed=True,
        bgcolor="white",
        overlap="false",
        splines="true",
        layout=layout,
    )

    if seed is not None:
        np.random.seed(seed)

    colors = {node.name: generate_random_color(colorscheme) for node in graph.nodes}

    for node in graph.nodes:
        g.add_node(
            node.name,
            fillcolor=colors[node.name],
            color="black",
            style="filled, rounded",
            shape="box",
            fontname="Helvetica",
        )

    scores = graph.get_all_edge_scores()
    max_score = max(scores)
    min_score = min(scores)

    for edge in graph.edges:
        score = graph.get_edge_score(edge)
        normalized_score = (
            (abs(score) - min_score) / (max_score - min_score)
            if max_score != min_score
            else abs(score)
        )
        penwidth = max(minimum_penwidth, normalized_score * maximum_penwidth)
        g.add_edge(
            edge.parent.name,
            edge.child.name,
            penwidth=str(penwidth),
            color=get_color(edge),
        )
    return g
