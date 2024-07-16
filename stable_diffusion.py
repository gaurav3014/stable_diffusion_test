import os
import streamlit as st
from PIL import Image
from torch import autocast
from diffusers import StableDiffusionPipeline
import torch

# Streamlit setup
st.set_page_config(page_title="Stable Bud", page_icon=":art:", layout="centered")
st.title("Stable Bud")

# User input for the prompt
prompt = st.text_input("Enter your prompt here:", "")

# Button to trigger image generation
if st.button("Generate"):
    if prompt:
        model_id = "CompVis/stable-diffusion-v1-4"
        device = "cpu"

        # Retrieve the token from environment variables
        auth_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not auth_token:
            st.error("API token not found. Please set the HUGGINGFACEHUB_API_TOKEN environment variable.")
        else:
            # Load the model
            pipe = StableDiffusionPipeline.from_pretrained(model_id, revision='fp16', torch_dtype=torch.float16, use_auth_token=auth_token)
            pipe.to(device)

            # Generate the image
            with autocast(device):
                image = pipe(prompt, guidance_scale=8.5)["sample"][0]
            
            # Save and display the image
            image.save("generated_image.png")
            st.image(image, caption="Generated Image")
    else:
        st.error("Please enter a prompt to generate an image.")
