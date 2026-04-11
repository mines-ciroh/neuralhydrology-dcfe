# demos/theme.py

import gradio as gr

NORD = {
    "bg":      "#2E3440",
    "panel":   "#3B4252",
    "panel2":  "#434C5E",
    "border":  "#4C566A",
    "blue":    "#5E81AC",
    "lblue":   "#81A1C1",
    "cyan":    "#88C0D0",
    "green":   "#A3BE8C",
    "red":     "#BF616A",
    "yellow":  "#EBCB8B",
    "text":    "#ECEFF4",
    "subtext": "#D8DEE9",
}

theme = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.slate,
    font=gr.themes.GoogleFont("Inter"),
).set(
    body_background_fill=NORD["bg"],
    body_text_color=NORD["text"],
    block_background_fill=NORD["panel"],
    block_border_color=NORD["border"],
    block_label_text_color=NORD["subtext"],
    button_primary_background_fill=NORD["blue"],
    button_primary_background_fill_hover=NORD["lblue"],
    button_primary_text_color=NORD["text"],
    input_background_fill=NORD["panel2"],
    input_border_color=NORD["border"],
    input_placeholder_color=NORD["border"],
)