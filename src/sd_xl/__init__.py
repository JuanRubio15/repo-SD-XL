import torch # Ejecutar operaciones matematicas del modelo de IA
from diffusers import AutoPipelineForText2Image
# Diffusers es una libreria especializada en modelos generativos

print("Cargando el modelo ...")

modelo = AutoPipelineForText2Image.from_pretrained(
     "stabilityai/sd-turbo",
      torch_dtype=torch.float32 #float16 o float32 torch.floatXX, indica la presiciÃ³n numerica con que se
                                #cargan los pesos para generar la imagen

)

modelo = modelo.to("cpu")

prompt = input("Escribe el prompt de la imagen que quieres crear: ")
#negative_prompt="worst quality, low quality, normal quality, low resolution, blurry, distorted, deformed, ugly, bad anatomy, bad proportions, bad hands, extra fingers, missing fingers, fused fingers, extra limbs, missing limbs, extra arms, extra legs, malformed limbs, twisted body, out of frame, cut off, duplicate, watermark, signature, logo, text, draft, artifact, pixelated"
print("Generando imagen...")

imagen = modelo(
    prompt = prompt, #Instrucciones o detalle para crear la imagen
    # negative_prompt=negative_prompt,
    num_inference_steps=25, #Las veces que va a crear o refinar la imagen
    guidance_scale=7.0,#Que tan fiel debe ser al prompt
    height=1024,
    width=1024
).images[0]

imagen.save("imagen.png")
print("Imagen guardada.")
