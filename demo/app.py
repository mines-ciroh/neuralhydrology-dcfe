import gradio as gr
from pages.streamflow_page import page as streamflow_page
from pages.internal_states_page import page as internal_states_page
from pages.parameters_page import page as parameters_page
from load_data import load_shm_data, load_lstm_data, load_dcfe_data

# Only load the data once, since loading is expensive.
shm_data = load_shm_data()
dcfe_data = load_dcfe_data()
lstm_data = load_lstm_data()

demo = gr.TabbedInterface(
    [streamflow_page(shm_data, dcfe_data, lstm_data), internal_states_page(shm_data, dcfe_data), parameters_page(shm_data, dcfe_data)],
    ["Streamflow", "Internal States", "Parameters"],
)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Monochrome(), share=True)
