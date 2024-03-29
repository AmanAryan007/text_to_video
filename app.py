import os
import torch
import streamlit as st
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video

# Initialize the pipeline
pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

st.title("Video Generator")

prompt = st.text_input("Enter a prompt:", "spiderman dancing")

if st.button("Generate Video"):
    video_frames = pipe(prompt, num_inference_steps=25).frames

    vid_name = 'vid'
    output_name = os.path.join(os.getcwd(), 'temp', f'{vid_name}.mp4')
    video_path = export_to_video(video_frames, output_video_path=output_name)

    st.write(f'Generated video: {video_path}')

    # Display a download button for the generated video
    st.download_button(
        label="Download video",
        data=open(output_name, 'rb').read(),
        file_name=f'{vid_name}.mp4',
        mime='video/mp4'
    )
