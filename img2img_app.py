import gradio as gr

negative_prompt_list = 'EasyNegative, worst quality, low quality, normal quality, child, painting, drawing, sketch, cartoon, anime, render, 3d, blurry, deformed, disfigured, morbid, mutated, bad anatomy, bad art'

def create_demo(process: function):
    gradio_image = gr.Image(source="upload", type="filepath", label="Raw Image. Must Be .png")
    prompt = gr.Textbox(label = 'Prompt Input Text. 77 Token (Keyword or Symbol) Maximum')
    negative_prompt = gr.Textbox(label='Negative Prompt', value=negative_prompt_list)
    guidance_scale = gr.Slider(2, 15, value = 7, label = 'Guidance Scale')
    number_of_iterations = gr.Slider(1, 50, value = 10, step = 1, label = 'Number of Iterations')
    seed = gr.Slider(label = "Seed", minimum = 0, maximum = 987654321987654321, step = 1, randomize = True)
    strength = gr.Slider(label='Strength', minimum = 0, maximum = 1, step = .05, value = .5)

    with gr.Blocks() as demo:
     with gr.Tab("Image2Image"):
        gr.Interface(
           fn=process,
           inputs=[
              gradio_image,
              prompt,
              negative_prompt,
              guidance_scale,
              number_of_iterations,
              seed,
              strength,
           ],
           outputs='image', 
           title = "Stable Diffusion 2.1 Image to Image line", 
           description = "This is a thing I built",
        )

    return demo

if __name__ == '__main__':
    from model import Model
    model = Model()
    demo = create_demo(model.run_img2img_pipe)
    demo.queue().launch()