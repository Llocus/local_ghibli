from typing import Optional
import spaces
import os
import torch
from PIL import Image, ImageDraw, ImageFont
import gradio as gr

from src.pipeline import FluxPipeline
from src.transformer_flux import FluxTransformer2DModel
from src.lora_helper import set_single_lora, set_multi_lora, unset_lora
from transformers import BitsAndBytesConfig
import argparse

parser = argparse.ArgumentParser(description='Ghibli Image Generation')
parser.add_argument('--load-in-4bit', action='store_true', help='Load model in 4-bit')
parser.add_argument('--load-in-8bit', action='store_true', help='Load model in 8-bit')
parser.add_argument('--listen', action='store_true', help='Run on public ip')

args = parser.parse_args()

if args.load_in_4bit and args.load_in_8bit:
    raise ValueError("Error: Choose only one mode - use --load-in-4bit OR --load-in-8bit")

if args.load_in_4bit:
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
    )
elif args.load_in_8bit:
    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )
else:
    quant_config = None

# Initialize the image processor
base_path = "black-forest-labs/FLUX.1-dev"    
lora_base_path = "./models"

# System prompt that will be hidden from users but automatically added to their input
SYSTEM_PROMPT = "Ghibli Studio style, Charming hand-drawn anime-style illustration"

pipe = FluxPipeline.from_pretrained(base_path, torch_dtype=torch.bfloat16)
transformer = FluxTransformer2DModel.from_pretrained(base_path, subfolder="transformer", quantization_config=quant_config, torch_dtype=torch.bfloat16)
pipe.transformer = transformer
pipe.to("cuda")

empty_img = Image.new("RGB", (640, 640), color = (255, 255, 255))

def clear_cache(transformer):
    for name, attn_processor in transformer.attn_processors.items():
        attn_processor.bank_kv.clear()

# Define the Gradio interface
@spaces.GPU()
def single_condition_generate_image(user_prompt: str, spatial_img: Optional[Image.Image] = None, height: int=512, width: int=512, seed: int=42):
    # Combine the system prompt with user prompt
    full_prompt = f"{SYSTEM_PROMPT}, {user_prompt}" if user_prompt else SYSTEM_PROMPT
    
    # Set the Ghibli LoRA
    lora_path = os.path.join(lora_base_path, "Ghibli.safetensors")
    set_single_lora(pipe.transformer, lora_path, lora_weights=[1], cond_size=512)
    
    # Process the image
    if not spatial_img:
        spatial_imgs = [empty_img]
    else:
        spatial_imgs = [spatial_img] if spatial_img else []
    image = pipe(
        full_prompt,
        height=int(height),
        width=int(width),
        guidance_scale=3.5,
        num_inference_steps=25,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(seed), 
        subject_images=[],
        spatial_images=spatial_imgs,
        cond_size=512,
    ).images[0]
    clear_cache(pipe.transformer)
    return image

# New function for multilingual text rendering
@spaces.GPU()
def text_rendering_generate_image(user_prompt: str, input_text: str, text_color: str, text_size: int, text_position: str, spatial_img: Optional[Image.Image] = None, height: int=512, width: int=512, seed: int=42):
    # Combine the system prompt with user prompt
    full_prompt = f"{SYSTEM_PROMPT}, {user_prompt}" if user_prompt else SYSTEM_PROMPT
    
    # Set the Ghibli LoRA
    lora_path = os.path.join(lora_base_path, "Ghibli.safetensors")
    set_single_lora(pipe.transformer, lora_path, lora_weights=[1], cond_size=512)
    
    # Process the image
    if not spatial_img:
        spatial_imgs = [empty_img]
    else:
        spatial_imgs = [spatial_img] if spatial_img else []
    image = pipe(
        full_prompt,
        height=int(height),
        width=int(width),
        guidance_scale=3.5,
        num_inference_steps=25,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(seed), 
        subject_images=[],
        spatial_images=spatial_imgs,
        cond_size=512,
    ).images[0]
    
    # Add text to the generated image if text is provided
    if input_text:
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        # Create a drawing context
        draw = ImageDraw.Draw(image)
        
        # Try to load a font that supports multilingual text
        try:
            # Attempt to load a system font that supports multilingual text
            # Scale up the text size significantly to make it more visible
            actual_text_size = text_size * 3  # Multiply the size by 3 for better visibility
            font = ImageFont.truetype("Arial Unicode.ttf", actual_text_size)
        except IOError:
            try:
                # Try another common font if Arial Unicode is not available
                actual_text_size = text_size * 3
                font = ImageFont.truetype("DejaVuSans.ttf", actual_text_size)
            except IOError:
                # Final fallback to default font with increased size
                font = ImageFont.load_default()
        
        # Parse position (top, center, bottom)
        # Use actual_text_size for position calculations to maintain proper spacing
        if text_position == "top":
            position = (width // 2, actual_text_size + 30)  # More padding from the top
        elif text_position == "bottom":
            position = (width // 2, height - actual_text_size - 30)  # More padding from the bottom
        else:  # center
            position = (width // 2, height // 2)
        
        # Add text with outline for better visibility
        # Draw text outline (shadow) with larger offset for better visibility
        outline_size = max(3, actual_text_size // 15)  # Scale outline size with text size
        for offset_x in range(-outline_size, outline_size + 1, outline_size):
            for offset_y in range(-outline_size, outline_size + 1, outline_size):
                if offset_x == 0 and offset_y == 0:
                    continue  # Skip the center position (will be drawn as main text)
                draw.text(
                    (position[0] + offset_x, position[1] + offset_y),
                    input_text,
                    fill="black",
                    font=font,
                    anchor="mm"  # Center align the text
                )
        
        # Draw the main text
        draw.text(
            position,
            input_text,
            fill=text_color,
            font=font,
            anchor="mm"  # Center align the text
        )
    
    clear_cache(pipe.transformer)
    return image

# Load example images
def load_examples():
    examples = []
    test_img_dir = "./test_imgs"
    example_prompts = [
        " ",
        "saying 'HELLO' in 'speech bubble'",
        "background 'alps'"
    ]
    
    for i, filename in enumerate(["00.jpg", "02.jpg", "03.jpg"]):
        img_path = os.path.join(test_img_dir, filename)
        if os.path.exists(img_path):
            # Use dimensions from original code for each specific example
            if filename == "00.jpg":
                height, width = 680, 1024
            elif filename == "02.jpg":
                height, width = 560, 1024
            elif filename == "03.jpg":
                height, width = 1024, 768
            else:
                height, width = 768, 768
                
            examples.append([
                example_prompts[i % len(example_prompts)],  # User prompt (without system prompt)
                Image.open(img_path),                       # Reference image
                height,                                     # Height
                width,                                      # Width
                i + 1                                       # Seed
            ])
    
    return examples

# Load examples for text rendering tab
def load_text_examples():
    examples = []
    test_img_dir = "./test_imgs"
    
    example_data = [
        {
            "prompt": "cute character with speech bubble",
            "text": "Hello World!",
            "color": "#ffffff",
            "size": 72,
            "position": "center",
            "filename": "00.jpg",
            "height": 680,
            "width": 1024,
            "seed": 123
        },
        {
            "prompt": "landscape with message",
            "text": "안녕하세요!",
            "color": "#ffff00",
            "size": 100,
            "position": "top",
            "filename": "03.jpg",
            "height": 1024,
            "width": 768,
            "seed": 456
        },
        {
            "prompt": "character with subtitles",
            "text": "こんにちは世界!",
            "color": "#00ffff",
            "size": 90,
            "position": "bottom",
            "filename": "02.jpg",
            "height": 560,
            "width": 1024,
            "seed": 789
        }
    ]
    
    for example in example_data:
        img_path = os.path.join(test_img_dir, example["filename"])
        if os.path.exists(img_path):
            examples.append([
                example["prompt"],
                example["text"],
                example["color"],
                example["size"],
                example["position"],
                Image.open(img_path),
                example["height"],
                example["width"],
                example["seed"]
            ])
    
    return examples

# CSS for improved UI
css = """
:root {
    --primary-color: #4a6670;
    --accent-color: #ff8a65;
    --background-color: #f5f5f5;
    --card-background: #f97316;
    --text-color: #333333;
    --border-radius: 10px;
    --shadow: 0 4px 6px rgba(0,0,0,0.1);
}

body {
    background-color: var(--background-color);
    color: var(--text-color);
    font-family: 'Helvetica Neue', Arial, sans-serif;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
}

.gr-header {
    background: linear-gradient(135deg, #668796 0%, #4a6670 100%);
    padding: 24px;
    border-radius: var(--border-radius);
    margin-bottom: 24px;
    box-shadow: var(--shadow);
    text-align: center;
}

.gr-header h1 {
    color: white;
    font-size: 2.5rem;
    margin: 0;
    font-weight: 700;
}

.gr-header p {
    color: rgba(255, 255, 255, 0.9);
    font-size: 1.1rem;
    margin-top: 8px;
}

.gr-panel {
    background-color: var(--card-background);
    border-radius: var(--border-radius);
    padding: 16px;
    box-shadow: var(--shadow);
}

.gr-button {
    background-color: var(--accent-color);
    border: none;
    color: white;
    padding: 10px 20px;
    border-radius: 5px;
    font-size: 16px;
    font-weight: bold;
    cursor: pointer;
    transition: transform 0.1s, background-color 0.3s;
}

.gr-button:hover {
    background-color: #ff7043;
    transform: translateY(-2px);
}

.gr-input, .gr-select {
    border-radius: 5px;
    border: 1px solid #ddd;
    padding: 10px;
    width: 100%;
}

.gr-form {
    display: grid;
    gap: 16px;
}

.gr-box {
    background-color: var(--card-background);
    border-radius: var(--border-radius);
    padding: 20px;
    box-shadow: var(--shadow);
    margin-bottom: 20px;
}

.gr-gallery {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 16px;
}

.gr-gallery-item {
    overflow: hidden;
    border-radius: var(--border-radius);
    box-shadow: var(--shadow);
    transition: transform 0.3s;
}

.gr-gallery-item:hover {
    transform: scale(1.02);
}

.gr-image {
    width: 100%;
    height: auto;
    object-fit: cover;
}

.gr-footer {
    text-align: center;
    margin-top: 40px;
    padding: 20px;
    color: #666;
    font-size: 14px;
}

.gr-examples-gallery {
    margin-top: 20px;
}

/* Responsive adjustments */
@media (max-width: 768px) {
    .gr-header h1 {
        font-size: 1.8rem;
    }
    
    .gr-panel {
        padding: 12px;
    }
}

/* Ghibli-inspired accent colors */
.gr-accent-1 {
    background-color: #95ccd9;
}

.gr-accent-2 {
    background-color: #74ad8c;
}

.gr-accent-3 {
    background-color: #f9c06b;
}

.text-rendering-options {
    background-color: #f0f8ff;
    padding: 16px;
    border-radius: var(--border-radius);
    margin-top: 16px;
}
"""

# Create the Gradio Blocks interface
with gr.Blocks(css=css) as demo:
    gr.HTML("""
    <div class="gr-header">
        <h1>✨ Ghibli Multilingual Text-Rendering ✨</h1>
        <p>Transform your ideas into magical Ghibli-inspired artwork</p>
    </div>
    """)
    
    with gr.Tabs():
        with gr.Tab("Create Ghibli Art"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML("""
                    <div class="gr-box">
                        <h3>🎨 Your Creative Input</h3>
                        <p>Describe what you want to see in your Ghibli-inspired image</p>
                    </div>
                    """)
                    
                    user_prompt = gr.Textbox(
                        label="Your description", 
                        placeholder="Describe what you want to see (e.g., a cat sitting by the window)",
                        lines=2
                    )
                    
                    spatial_img = gr.Image(
                        label="Reference Image (Optional)", 
                        type="pil",
                        elem_classes="gr-image-upload"
                    )
                    
                    with gr.Group():
                        with gr.Row():
                            height = gr.Slider(minimum=256, maximum=1024, step=64, label="Height", value=768)
                            width = gr.Slider(minimum=256, maximum=1024, step=64, label="Width", value=768)
                        
                        seed = gr.Slider(minimum=1, maximum=9999, step=1, label="Seed", value=42, 
                                        info="Change for different variations")
                    
                    generate_btn = gr.Button("✨ Generate Ghibli Art", elem_classes="gr-button")
                
                with gr.Column(scale=1):
                    gr.HTML("""
                    <div class="gr-box">
                        <h3>✨ Your Magical Creation</h3>
                        <p>Your Ghibli-inspired artwork will appear here</p>
                    </div>
                    """)
                    output_image = gr.Image(label="Generated Image", elem_classes="gr-output-image")
            
            gr.HTML("""
            <div class="gr-box gr-examples-gallery">
                <h3>✨ Inspiration Gallery</h3>
                <p>Click on any example to try it out</p>
            </div>
            """)
            
            # Add examples
            examples = load_examples()
            gr.Examples(
                examples=examples,
                inputs=[user_prompt, spatial_img, height, width, seed],
                outputs=output_image,
                fn=single_condition_generate_image,
                cache_examples=False,
                examples_per_page=4
            )
            
            # Link the button to the function
            generate_btn.click(
                single_condition_generate_image,
                inputs=[user_prompt, spatial_img, height, width, seed],
                outputs=output_image
            )
        
        # Second tab for Image & Multilingual Text Rendering
        with gr.Tab("Image & Multilingual Text Rendering"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML("""
                    <div class="gr-box">
                        <h3>🌈 Art with Text</h3>
                        <p>Create Ghibli-style images with beautiful text in any language</p>
                    </div>
                    """)
                    
                    text_user_prompt = gr.Textbox(
                        label="Image Description", 
                        placeholder="Describe what you want to see (e.g., a character with speech bubble)",
                        lines=2
                    )
                    
                    with gr.Group(elem_classes="text-rendering-options"):
                        input_text = gr.Textbox(
                            label="Overlay Text", 
                            placeholder="Enter text in any language",
                            lines=1
                        )
                        
                        with gr.Row():
                            text_color = gr.ColorPicker(
                                label="Text Color", 
                                value="#FFFFFF"
                            )
                            
                            text_size = gr.Slider(
                                minimum=24, 
                                maximum=200, 
                                step=4, 
                                label="Text Size", 
                                value=72
                            )
                        
                        text_position = gr.Radio(
                            ["top", "center", "bottom"], 
                            label="Text Position", 
                            value="center"
                        )
                    
                    text_spatial_img = gr.Image(
                        label="Reference Image (Optional)", 
                        type="pil",
                        elem_classes="gr-image-upload"
                    )
                    
                    with gr.Group():
                        with gr.Row():
                            text_height = gr.Slider(minimum=256, maximum=1024, step=64, label="Height", value=768)
                            text_width = gr.Slider(minimum=256, maximum=1024, step=64, label="Width", value=768)
                        
                        text_seed = gr.Slider(minimum=1, maximum=9999, step=1, label="Seed", value=42, 
                                           info="Change for different variations")
                    
                    text_generate_btn = gr.Button("✨ Generate Art with Text", elem_classes="gr-button")
                
                with gr.Column(scale=1):
                    gr.HTML("""
                    <div class="gr-box">
                        <h3>✨ Your Text Creation</h3>
                        <p>Your Ghibli-inspired artwork with text will appear here</p>
                    </div>
                    """)
                    text_output_image = gr.Image(label="Generated Image with Text", elem_classes="gr-output-image")
            
            gr.HTML("""
            <div class="gr-box gr-examples-gallery">
                <h3>✨ Text Rendering Examples</h3>
                <p>Click on any example to try it out</p>
            </div>
            """)
            
            # Add text rendering examples
            text_examples = load_text_examples()
            gr.Examples(
                examples=text_examples,
                inputs=[text_user_prompt, input_text, text_color, text_size, text_position, 
                        text_spatial_img, text_height, text_width, text_seed],
                outputs=text_output_image,
                fn=text_rendering_generate_image,
                cache_examples=False,
                examples_per_page=3
            )
            
            # Link the text render button to the function
            text_generate_btn.click(
                text_rendering_generate_image,
                inputs=[text_user_prompt, input_text, text_color, text_size, text_position, 
                        text_spatial_img, text_height, text_width, text_seed],
                outputs=text_output_image
            )
    
    gr.HTML("""
    <div class="gr-footer">
        <p>Powered by FLUX.1 and Ghibli LoRA • Created with ❤️</p>
    </div>
    """)

# Launch the Gradio app
demo.queue().launch(server_name="0.0.0.0" if args.listen else "127.0.0.1")