import gradio as gr
import threading
import os
import torch

os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
torch.set_num_threads(os.cpu_count())

model1 = gr.load("models/prithivMLmods/SD3.5-Turbo-Realism-2.0-LoRA")
model2 = gr.load("models/Purz/face-projection")

stop_event = threading.Event()

def generate_images(text, selected_model):
    stop_event.clear()

    if selected_model == "Model 1 (Turbo Realism)":
        model = model1
    elif selected_model == "Model 2 (Face Projection)":
        model = model2
    else:
        return ["Invalid model selection."] * 3

    results = []
    for i in range(3):
        if stop_event.is_set():
            return ["Image generation stopped by user."] * 3

        modified_text = f"{text} variation {i+1}"
        result = model(modified_text)
        results.append(result)

    return results

def stop_generation():
    """Stops the ongoing image generation by setting the stop_event flag."""
    stop_event.set()
    return ["Generation stopped."] * 3

with gr.Blocks() as interface:#...
    gr.Markdown(
        "### ⚠ Sorry for the inconvenience. The Space is currently running on the CPU, which might affect performance. We appreciate your understanding."
    )
    
    text_input = gr.Textbox(label="Type here your imagination:", placeholder="Type your prompt...")
    model_selector = gr.Radio(
        ["Model 1 (Turbo Realism)", "Model 2 (Face Projection)"],
        label="Select Model",
        value="Model 1 (Turbo Realism)"
    )
    
    with gr.Row():
        generate_button = gr.Button("Generate 3 Images 🎨")
        stop_button = gr.Button("Stop Image Generation")
    
    with gr.Row():
        output1 = gr.Image(label="Generated Image 1")
        output2 = gr.Image(label="Generated Image 2")
        output3 = gr.Image(label="Generated Image 3")
    
    generate_button.click(generate_images, inputs=[text_input, model_selector], outputs=[output1, output2, output3])
    stop_button.click(stop_generation, inputs=[], outputs=[output1, output2, output3])

interface.launch()