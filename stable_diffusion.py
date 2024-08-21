import os
import streamlit as st
from PIL import Image
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
        device = "cpu"  # Change to "cuda" if you have a GPU

        # Retrieve the token from environment variables
        auth_token = st.secrets.get("API_TOKEN")
        if not auth_token:
            st.error("API token not found. Please set the API_TOKEN in Streamlit secrets.")
        else:
            # Load the model
            pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=auth_token)
            pipe.to(device)

            # Generate the image
            with torch.no_grad():  # No need for autocast on CPU
                image = pipe(prompt, guidance_scale=8.5)["sample"][0]
            
            # Save and display the image
            image.save("generated_image.png")
            st.image(image, caption="Generated Image")
    else:
        st.error("Please enter a prompt to generate an image.")
