from PIL import Image
import os

def convert_to_jpg(input_directory, output_directory):
    # Si el directorio de salida no existe, créalo
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Recorre todos los archivos en el directorio de entrada
    for filename in os.listdir(input_directory):
        if filename.endswith((".png", ".jpeg", ".gif", ".bmp", ".tiff", ".webp")): # Agrega o elimina las extensiones de archivo según sea necesario
            # Crea la ruta de archivo completa y carga la imagen
            input_path = os.path.join(input_directory, filename)
            image = Image.open(input_path)

            # Convertir la imagen a RGB si es necesario
            if image.mode not in ("RGB", "RGBA"):
                image = image.convert("RGB")

            # Guardar en el nuevo formato
            output_path = os.path.join(output_directory, os.path.splitext(filename)[0] + ".jpg")
            image.save(output_path, "JPEG")

            print(f"Imagen convertida: {input_path} -> {output_path}")

    print("Conversión completada.")

# Uso
input_directory = r"C:\Users\benit\OneDrive\Desktop\Reconocimiento de caras\Benito Fotos Randoms"  # directorio de las imágenes de origen
output_directory = r"C:\Users\benit\OneDrive\Desktop\Reconocimiento de caras\Benito Fotos Randoms\Converted"  # directorio para guardar las imágenes convertidas
convert_to_jpg(input_directory, output_directory)
