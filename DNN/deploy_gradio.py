import gradio as gr
from inference_gradio import model_SemEval, model_davidson, process_input
import numpy as np
import os
def process_output_sem(predictions):
    if np.argmax(np.array(predictions)):
        return "Hate speech"
    else:
        return "Non-hate speech"
def process_output_dad(predictions):
    if np.argmax(np.array(predictions)):
        return "Non-hate speech"
    else:
        return "Hate speech"
def process_text(input_text, dataset):
    if dataset == "SemEval":
        input = process_input(input_text, vocab_path= os.path.join(path_model_SemEval, "vocab.pkl"))
        predictions = SemEval(input)
        return process_output_sem(predictions)
    else:
        input = process_input(input_text, vocab_path= os.path.join(path_model_davidson, "vocab.pkl"))
        predictions = davidson(input)
        return process_output_dad(predictions)

demo = gr.Interface(
    fn=process_text,
    inputs=[
        gr.Textbox("", label="Input text", placeholder="Type your input here..."),
        gr.Radio(["SemEval", "Davidson"], value='SemEval', label="Model trained with dataset...", type="value")
    ],
    outputs=[gr.Textbox(None, label="Result")],
)

if __name__ == "__main__":
    path_model_SemEval = "outputs/SemEval" # path to model semEval
    path_model_davidson = "outputs/davidson/final" # path to model davidson
    SemEval = model_SemEval(os.path.join(path_model_SemEval, "best_model"))
    davidson = model_davidson(os.path.join(path_model_davidson, "best_model"))
    demo.launch(share=True)
