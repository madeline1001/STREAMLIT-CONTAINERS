import streamlit as st
import pandas as pd
import pickle
from pycaret.regression import predict_model


# Cargar el modelo y los datos en st.session_state si no están ya cargados
if 'modelo' not in st.session_state:
    model_path = "best_model.pkl"
    with open('best_model.pkl', 'rb') as model_file:
        st.session_state['modelo'] = pickle.load(model_file)

if 'test_data' not in st.session_state:

    prueba_ = pd.read_csv('prueba_APP.csv', header=0, sep=";", decimal=",")
    prueba_.drop(columns=['Email', 'Address', 'price'], inplace=True)
    
    st.session_state['test_data'] = prueba_


# Función para predicción individual
def prediccion_individual():
    st.header("Predicción manual de datos")
    
    # Inputs manuales
    dominio = st.selectbox("Seleccione el dominio:", ['yahoo','gmail', 'Otro', 'hotmail'])
    Tec = st.selectbox("Seleccione el tipo de dispositivo:", ['PC','Smartphone', 'Portatil', 'Iphone'])
    Address = st.selectbox("Seleccione la dirección:", ['Munich', 'Ausburgo', 'Berlin', 'Frankfurt'])
    Avg = st.text_input("Ingrese Avg. Session Length:", value="33.946239")
    Time_App = st.text_input("Ingrese Time on App:", value="10.983977")
    Time_Web = st.text_input("Ingrese Time on Website:", value="37.951488")
    Length = st.text_input("Ingrese Length of Membership:", value="3.050713")

    if st.button("Calcular predicción manual"):
        # Crear el dataframe de los inputs
        user = pd.DataFrame({'x0': [dominio], 'x1': [Tec], 'x2': [Avg], 'x3':[Time_App],
            'x4':[Time_Web], 'x5': [Length]})

        
        # Cargar los datos de prueba y concatenar
        prueba_ = st.session_state['test_data']
        user.columns = prueba_.columns
        prueba2_ = pd.concat([user, prueba_], axis=0)
        prueba2_.index = range(prueba2_.shape[0])

        # Hacer predicciones
        predictions = predict_model(st.session_state['modelo'], data=prueba2_)

        predictions["price"] = predictions["prediction_label"]

        st.write(f'La predicción es: {predictions.iloc[0]["price"]}')

    if st.button("Volver al menú principal"):
        st.session_state['menu'] = 'main'


prueba = pd.read_csv("prueba_APP.csv",header = 0,sep=";",decimal=",")
# Cargar el modelo preentrenado
with open('best_model.pkl', 'rb') as model_file:
    modelo = pickle.load(model_file)


# Función para predicción por base de datos
def prediccion_base_datos():
    st.header("Cargar archivo para predecir")
    uploaded_file = st.file_uploader("Cargar archivo Excel o CSV", type=["xlsx", "csv"])

    if st.button("Predecir con archivo"):
        if uploaded_file is not None:
            try:
                # Cargar el archivo directamente sin usar tempfile
                if uploaded_file.name.endswith(".csv"):
                    prueba = pd.read_csv(uploaded_file,header = 0,sep=";",decimal=",")
                else:
                    prueba = pd.read_excel(uploaded_file,header = 0,sep=";",decimal=",")

                # Realizar predicción
                predictions = predict_model(modelo, data=prueba)
                predictions["price"] = predictions["prediction_label"]

                # Preparar archivo para descargar
                kaggle = pd.DataFrame({'Email': prueba["Email"], 'price': predictions["price"]})

                # Mostrar predicciones en pantalla
                st.write("Predicciones generadas correctamente!")
                st.write(kaggle)

                # Botón para descargar el archivo de predicciones
                st.download_button(label="Descargar archivo de predicciones",
                               data=kaggle.to_csv(index=False),
                               file_name="kaggle_predictions.csv",
                               mime="text/csv")

            except Exception as e:
                st.error(f"Error: {e}")
        
        else:
            st.error("Por favor, cargue un archivo válido.")
    
    if st.button("Volver al menú principal"):
        st.session_state['menu'] = 'main'


# Función principal para mostrar el menú de opciones
def menu_principal():
    st.title("API de Predicción del precio")
    option = st.selectbox("Seleccione una opción", ["", "Predicción Individual", "Predicción Base de Datos"])

    if option == "Predicción Individual":
        st.session_state['menu'] = 'individual'
    elif option == "Predicción Base de Datos":
        st.session_state['menu'] = 'base_datos'

# Lógica para manejar el flujo de la aplicación
if 'menu' not in st.session_state:
    st.session_state['menu'] = 'main'

if st.session_state['menu'] == 'main':
    menu_principal()
elif st.session_state['menu'] == 'individual':
    prediccion_individual()
elif st.session_state['menu'] == 'base_datos':
    prediccion_base_datos()
