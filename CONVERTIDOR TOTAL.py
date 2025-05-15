# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 23:56:47 2025

@author: Azul8
"""
##DICOM A NIFTY
import SimpleITK as sitk
import os

# Ruta a la carpeta con los DICOMs (una serie completa)
dicom_folder = 'C://Users//Azul8//OneDrive//Escritorio//unet imagenes//Rosalba//10000184'  # Reemplaza con tu ruta de tomografia

# Leer la serie DICOM
reader = sitk.ImageSeriesReader()
dicom_names = reader.GetGDCMSeriesFileNames(dicom_folder)
reader.SetFileNames(dicom_names)

# Convertir a imagen
image = reader.Execute()

# Guardar como NIfTI (.nii)
output_path = 'C://Users//Azul8//OneDrive//Escritorio//unet imagenes//archivos nifty//imagen_convertida.nii'  # sin .gz
sitk.WriteImage(image, output_path)

print(f'Archivo NIfTI (.nii) guardado en: {output_path}')
############################################################################################
#NII A NUMPY
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

# Rutas a las imágenes y máscaras
TC_path = 'C://Users//Azul8//OneDrive//Escritorio//unet imagenes//archivos nifty//imagen_convertida.nii'

def load_volume(path):
    """
    Cargar un volumen de imagen o máscara desde un archivo NIfTI.
    """
    nii = nib.load(path)
    volume = nii.get_fdata()
    return volume


def resize_volume(volume, new_shape=(128, 128, 128)):
    """
    Redimensionar un volumen de imagen a new_shape.
    Se puede ajustar esta función para mantener la relación de aspecto o utilizar interpolación.
    """
    # Calcular factores de zoom
    zh, zw, zd = np.array(new_shape) / np.array(volume.shape)
    # Aplicar zoom (simplificado, considerar ajustar la interpolación según sea necesario)
    resized_volume = zoom(volume, (zh, zw, zd), order=1) # order=1 (bilinear) es generalmente suficiente
    return resized_volume

# Cargar volúmenes
image_volume = load_volume(TC_path)


# Redimensionar volúmenes
image_resized = resize_volume(image_volume)

# Expandir las dimensiones para cumplir con la expectativa de entrada del modelo (añadiendo un eje de canal al final)
image_resized = np.expand_dims(image_resized, axis=-1)

np.save('C://Users//Azul8//OneDrive//Escritorio//unet imagenes//archivos numpy//imagen_3001.npy', image_resized)
##########################################################################
###APLICAR EL MODELO
from tensorflow.keras.models import load_model
my_model = load_model('aa.h5', compile=False)


test_img_input = np.expand_dims(image_resized, axis=0)
test_prediction = my_model.predict(test_img_input)
test_prediction = 1-test_prediction 

predicted_mask_binary = (test_prediction > 0.5).astype(bool)

################################
tr_path ='C://Users//Azul8//OneDrive//Escritorio//unet imagenes//archivos numpy//imagen_3001.npy'

# Cargar los archivos .npy como matrices NumPy
tr = np.load(tr_path)
import matplotlib.pyplot as plt

# Seleccionar una rebanada (slice) aleatoria
#n_slice = np.random.randint(0, tr.shape[2])
n_slice =100


# Extraer slice de imagen original
img_slice = tr[:, :, n_slice, 0]  # tr es (128,128,128,1)

# Extraer slice de la predicción
pred_slice = test_prediction[0, :, :, n_slice, 0]  # test_prediction es (1,128,128,128,1)

# Mostrar las imágenes
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(img_slice, cmap='gray')
plt.title('Imagen en TC')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(pred_slice, cmap='binary')
plt.title('Amplificacion del LL')
plt.axis('off')

plt.tight_layout()
plt.show()
################################################################
#CONVERTIR DE VUELTA A NUMPY Y SU RESOLUCION ORIGINAL

import numpy as np
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom

# Ruta al archivo original NIfTI
original_nii_path = 'C://Users//Azul8//OneDrive//Escritorio//unet imagenes//archivos nifty//imagen_convertida.nii'
original_nii = nib.load(original_nii_path)
original_shape = original_nii.shape
original_affine = original_nii.affine

# Ruta al archivo .npy (predicción o imagen redimensionada)
npy_path = 'C://Users//Azul8//OneDrive//Escritorio//unet imagenes//archivos numpy//imagen_3001.npy'
npy_array = np.load(npy_path)

# Eliminar el canal si existe (de forma (128,128,128,1) a (128,128,128))
if npy_array.shape[-1] == 1:
    npy_array = np.squeeze(npy_array, axis=-1)

# Redimensionar de vuelta al tamaño original
def resize_to_original(volume, target_shape):
    factors = np.array(target_shape) / np.array(volume.shape)
    return zoom(volume, factors, order=1)  # Interpolación bilineal

resized_to_original = resize_to_original(npy_array, original_shape)

# Crear nuevo objeto Nifti con la matriz redimensionada y el affine original
new_nii = nib.Nifti1Image(resized_to_original, affine=original_affine)

# Guardar el nuevo archivo NIfTI
output_path = 'C://Users//Azul8//OneDrive//Escritorio//unet imagenes//archivos nifty//prediccion_restaurada.nii'
nib.save(new_nii, output_path)

print(f"Archivo NIfTI restaurado guardado en: {output_path}")


 
