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
demo = gr.Interface(
    fn=process_text,
    inputs=[
        gr.Textbox("Text"),
        gr.Radio(["SemEval", "Davidson"], label="dataset", type="value")
    ],
    outputs="text"
)
if __name__ == "__main__":
    demo.launch(share=True)