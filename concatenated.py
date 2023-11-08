import tkinter as tk
import customtkinter as ctk
from diffusers import StableDiffusionPipeline

from PIL import ImageTk
import torch
import keras_cv
from tensorflow import keras

token = "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"

app = tk.Tk()

app.title("Styled Image Generator")
app.geometry("532x632")
ctk.set_appearance_mode("dark")

label = tk.Label(text="Input Image Generation Prompt:")
entry = ctk.CTkEntry(master=app, width=512, height=40, corner_radius=10, fg_color="white")

label.pack()
entry.pack()

prompt = entry.get()
prompt.place()

lmain = ctk.CTkLabel(master=app, height=512, width=512)
lmain.place(x=10, y=110) 
modelid = "CompVis/stable-diffusion-v1-4"
device = "cpu"
pipe = StableDiffusionPipeline.from_pretrained(modelid, revision="fp16", torch_dtype=torch.float16, use_auth_token=token) 
pipe.to(device) 

def generate(prompt): 
    
    image = pipe(prompt.get(), guidance_scale=8.5)["sample"][0]
    
    image.save('generatedimage.png')
    img = ImageTk.PhotoImage(image)
    lmain.configure(image=img)

trigger = ctk.CTkButton(master=app, height=25, width=100, text="CTkButton", fg_color="blue", command=generate) 
trigger.configure(text="Generate") 
trigger.place(x=217, y=60)

app.mainloop()