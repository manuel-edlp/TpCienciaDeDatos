# Tp Ciencia De Datos
repositorio del Tp Ciencia De Datos

## antes de arrancar:
instalar python, 
        git-bash (Requiere reiniciar compu), 
        ipynb                   


## Creacion de entorno virtual:
Para crear entorno virtual:
py -3.13 -m venv .venv   (cambiar por tu versión)  

Para activarlo por powershell:
.venv\Scripts\Activate.ps1

## Instalacion de dependencias
Una vez activo el entorno virtual instalar dependencias:
pip install -r requirements.txt                                                          

## Selección de entorno virtual como Kernel de la notebook
IMPORTANTE: AL EJECUTAR LAN NOTEBOOK SELECCIONAR COMO Kernel EL ENTORNO VIRTUAL CREADO. 
El Kernel seleccionado debería verse así por ejemplo: .venv (Python 3.13.3)
No elegir solo Python 3.13.3 porque no estarías usando el entorno virtual
        

streamlit run app.py