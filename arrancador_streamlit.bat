@echo off

rem Activamos el entorno base de Anaconda
call "C:\Users\iblan\anaconda3\Scripts\activate.bat"

rem Cambiamos al directorio de tu proyecto
cd /d C:\Users\iblan\Desktop\Proyecto_final

rem Ejecutamos tu aplicación Streamlit
streamlit run Home.py

pause