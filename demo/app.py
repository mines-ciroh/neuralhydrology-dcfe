import gradio as gr
from theme import theme
from pages.streamflow import page as streamflow_page
from pages.internal_states_and_params import page as internal_states_and_params_page

demo = gr.TabbedInterface(
    [streamflow_page, internal_states_and_params_page],
    ["Streamflow", "States & Parameters"],
)

if __name__ == "__main__":
    demo.launch(theme=theme)