import gradio as gr
from inference_gradio import model_SemEval, model_davidson

def process_text(input_text, dataset):
    if dataset == "SemEval":
        output = model_SemEval(input_text)
    elif dataset == "Davidson":
        output = model_davidson(input_text)
    else:
        output = "Invalid option selected"
    return output

# Choose a theme (e.g., "light", "dark", "gr", "sketch", "translucent", etc.)
theme = "sketch"

demo = gr.Interface(
    fn=process_text,
    inputs=[
        gr.Textbox("", label="Input text", placeholder="Type your input here..."),
        gr.Radio(["SemEval", "Davidson"], label="Model trained with dataset...", type="value")
    ],
    outputs=[gr.Textbox(None, label="Result")],
    theme=theme  # Add the theme parameter here
)

if __name__ == "__main__":
    demo.launch(share=True)
