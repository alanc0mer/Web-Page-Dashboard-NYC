import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit.components.v1 as components
import random
import os
import warnings
warnings.filterwarnings('ignore')
from streamlit_option_menu import option_menu
import folium
from streamlit_folium import st_folium
import os
from streamlit_extras.stylable_container import stylable_container
import plotly.graph_objects as go
import holidays
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import time

#@st.cache_resource

#Título--------------------------------------------------------------------------------------------------------------------------------------------------------

st.set_page_config(page_title="Prototipo", page_icon=":bar_chart:", layout="wide")
st.markdown('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)

#HTML/CSS----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# --- Ocultar basura de streamlit ---
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# Contenedores

def container_stylestxt(texto):
  with stylable_container(

        key="styConttxt",
        css_styles="""
            {
                background-color: #FFFFFF;
                border: 1px solid #CCCCCC;
                padding: 5% 5% 5% 10%;
                border-radius: 5px;

                border-left: 0.5rem solid #9AD8E1 !important;
                box-shadow: 0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.15) !important;
            }
            """,
    ):
        st.markdown(texto)

def container_styles_plt(fig):
  with stylable_container(

        key="styContplt",
        css_styles="""
            {
                background-color: #FFFFFF;
                border: 1px solid #CCCCCC;
                padding: 1% 5% 3% 1%;
                border-radius: 5px;

                border-left: 0.5rem solid #9AD8E1 !important;
                box-shadow: 0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.15) !important;
            }
            """,
    ):
        st.plotly_chart(aplicarFormatoChart(fig))


def container_stylesNumDuros(texto,num):
  with stylable_container(

        key="styContnum",
        css_styles="""
            {
                background-color: #FFFFFF;
                border: 1px solid #CCCCCC;
                padding: 5% 5% 5% 10%;
                border-radius: 5px;

                border-left: 0.5rem solid #9AD8E1 !important;
                box-shadow: 0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.15) !important;
            }
            """,
    ):
        st.metric(texto,num)



#Estilo

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
#Lectura de Bases de Datos--------------------------------------------------------------------------------------------------------------------------------------------------------

@st.cache_data()  # 👈 Add the caching decorator
def load_data(url):
    df = pd.read_csv(url)
    return df
df_ventas = load_data('VentasLimpio3.csv')

@st.cache_data  # 👈 Add the caching decorator
def load_data_json(url):
    df = url
    return df

#Funcion mapa----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def displaymapcount(df):
  map = folium.Map(location=[40.7128, -73.9], zoom_start=10, scrollWheelZoom=False, tiles= 'CartoDB positron')
  choropleth = folium.Choropleth(

      geo_data=load_data_json('Borough Boundaries.geojson'),
      data=df['County'].value_counts().reset_index(),
      columns=('County', 'count'),
      key_on='feature.properties.boro_name',
      line_opacity=0.7,
      highlight=True
  )
  choropleth.add_to(map)
  for feature in choropleth.geojson.data['features']:

      # Obtener Nombre de la lista Geojson
      boro_name = feature['properties']['boro_name']

      # Insertar en la lista Geojson, el conteo
      feature['properties']['count'] = 'Conteo: ' + str('{:,}'.format(df['County'].value_counts().get(boro_name, 0)))
      feature['properties']['price'] = 'Precio de venta: ' + str('${:,.0f}'.format(df_ventas_filtered.groupby('County')['SALE PRICE'].mean().get(boro_name, 0)))
      feature['properties']['ft2price'] = 'Ft2 de venta: ' + str('${:,.0f}'.format(df_ventas_filtered.groupby('County')['Precio_por_pie_cuadrado'].mean().get(boro_name, 0)))
      

  choropleth.geojson.add_child(
      folium.features.GeoJsonTooltip(['boro_name', 'count', 'price', 'ft2price'], labels=False)
  ) 
  return st_folium(map, width=700, height=500, returned_objects=[])


#Funcion Cache ML----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

@st.cache_resource  # 👈 Add the caching decorator
def prophet(df):
    
  holiday = pd.DataFrame([])

  us_holidays = holidays.XNYS()

  for date_, name in sorted(holidays.XNYS(years=[2019,2020,2021,2022,2023]).items()):
    holiday = pd.concat([holiday, pd.DataFrame({'ds': date_, 'holiday': "US-Holidays", 'lower_window': -1, 'upper_window': 1}, index=[0])], ignore_index=True)

  holiday['ds'] = pd.to_datetime(holiday['ds'], format='%Y-%m-%d', errors='ignore')
  
  df_ts=df.copy()
  df_ts.columns=['ds','y']
  df_ts['cap']=700
  df_ts['floor']=100
  m=Prophet(holidays=holiday)
  m.fit(df_ts)

  return m

@st.cache_resource  # 👈 Add the caching decorator
def forecast(hasta,frec,_m):
  future=m.make_future_dataframe(periods=hasta, freq=frec)
  future['cap']=700
  future['floor']=100
  forecast=m.predict(future)  
  return forecast

#Funcion Acomodo plot----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def aplicarFormatoChart(fig,controls=False,legend=True,hoverTemplate=None):
    fig.update_layout(paper_bgcolor='rgba(0, 0, 0, 0)')
    fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)')
    fig.update_layout(showlegend=legend)

    if hoverTemplate:
        if hoverTemplate=="%":
            fig.update_traces(hovertemplate='<b>%{x}</b> <br> %{y:,.2%}')
        elif hoverTemplate=="$":
            fig.update_traces(hovertemplate='<b>%{x}</b> <br> $ %{y:,.1f}')
        elif hoverTemplate=="#":
            fig.update_traces(hovertemplate='<b>%{x}</b> <br> %{y:,.0f}')
    if controls:
        fig.update_xaxes(
            rangeslider_visible=True
        )

    fig.update_layout(
            autosize=True,
            margin=dict(
                l=50,
                r=50,
                b=50,
                t=50,
                pad=4
            )
    )
    return fig


#Limpieza------------------------------------------------------------------------------------------------------------------------------------------------------------------------



@st.cache_data
def df_limpio(df_ventas):
    #Se crearon columnas inutiles en el anterior entregable accidentalmente ('Unnamed: 0' y 'Unnamed: 0.1'), por lo que se eliminaron.
    
    df_ventas.drop(columns=['Unnamed: 0','Unnamed: 0.1','Unnamed: 0.2'],inplace=True)
    
    #Definir SALE DATE como fecha
    
    df_ventas['SALE DATE'] = pd.to_datetime(df_ventas['SALE DATE'])
    
    #Crear compatibilidad con folium
    
    df_ventas['County']=df_ventas['County'].str.capitalize()
    df_ventas['County']=df_ventas['County'].str.replace('Staten_island', 'Staten Island')
    df_ventas['NEIGHBORHOOD'] = df_ventas['NEIGHBORHOOD'].apply(lambda x: x.strip())

    return df_ventas

df_ventas=df_limpio(df_ventas)

@st.cache_data
def df_queens(df_ventas,frec):
    #Base de datos de Queens
        
    #Solo Queens (del 2019 al 2023) 
    
    df_ventas_queens=df_ventas.loc[(df_ventas['County']=='Queens') & ((df_ventas['SALE DATE'].dt.year == 2019)|(df_ventas['SALE DATE'].dt.year == 2020)|(df_ventas['SALE DATE'].dt.year == 2021)|(df_ventas['SALE DATE'].dt.year == 2022)|(df_ventas['SALE DATE'].dt.year == 2023))]
    
    #Solo viviendas tipo A, B y C
    
    df_ventas_queens=df_ventas_queens.loc[df_ventas_queens['BUILDING CLASS AT PRESENT'].str.contains('A') | df_ventas_queens['BUILDING CLASS AT PRESENT'].str.contains('B') | df_ventas_queens['BUILDING CLASS AT PRESENT'].str.contains('C')]

    #Solo Neighborhood elegidos

    df_ventas_queens=df_ventas_queens.loc[df_ventas_queens['NEIGHBORHOOD'].isin(["SPRINGFIELD GARDENS", "SOUTH OZONE PARK", "HOLLIS", "CORONA", "GLENDALE", "EAST ELMHURST"])]

    #Solo Variables de estudio
    
    df_ventas_queens=df_ventas_queens[['SALE DATE','Precio_por_pie_cuadrado']]
    
    #Intervalo de confianza del 95%
    
    df_ventas_queens=df_ventas_queens.loc[(df_ventas_queens['Precio_por_pie_cuadrado']>df_ventas_queens['Precio_por_pie_cuadrado'].mean()-1.98*df_ventas_queens['Precio_por_pie_cuadrado'].std()) & (df_ventas_queens['Precio_por_pie_cuadrado'] < df_ventas_queens['Precio_por_pie_cuadrado'].mean()+1.98*df_ventas_queens['Precio_por_pie_cuadrado'].std())]
    
    #Ordenado por fecha

    df_ventas_queens.sort_values(by='SALE DATE', inplace=True)

    #Cambio de frecuencia al elegido

    df_ventas_queens=df_ventas_queens.resample(frec, on='SALE DATE').mean().reset_index()
    
    return df_ventas_queens


#Menu-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

selected = option_menu(

    menu_title=None,
    options=["Historia de NYC", "Dashboard de Queens", "Dashboard de usuario", "Pronóstico de Queens","Base de Datos"],
    icons=["clock-history", "building-check", "building-add", "graph-up-arrow", "download"],  # https://icons.getbootstrap.com/
    orientation="horizontal",
)

  #Select Box Historia NYC----------------------------------------------------------------------------------------------------------------------------------------------------------------------


if selected == "Historia de NYC":
  st.title("Ventas de Propiedades en la Ciudad de Nueva York")
  #Filtros opciones-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

  Year=df_ventas['SALE DATE'].dt.year.sort_values(ascending=True).unique()
  county=df_ventas['County'].sort_values(ascending=True).unique()    
  st.sidebar.header("Filtros")

  # Select Box Año Historia
  year_option = st.sidebar.multiselect(
      "Año",
      Year
  )

  if not year_option:
    year_option=df_ventas['SALE DATE'].dt.year.unique()


  # Select Box condado Historia
  county_option = st.sidebar.multiselect(
      "Condado",
      county
  )

  if not county_option:
    county_option=df_ventas['County'].unique()

  #Filtro en bases de datos Historia NYC--------------------------------------------------------------------------------------------------------------------------------------------------------------------------

  df_ventas_filtered=df_ventas[(df_ventas['County'].isin(county_option)) & (df_ventas['SALE DATE'].dt.year.isin(year_option))]

  # Precio por pie cuadrado promedio de Condados

  df_preciopromedio_anual=df_ventas_filtered[['County','SALE DATE','Precio_por_pie_cuadrado']].copy()
  df_preciopromedio_anual['SALE DATE'] = df_preciopromedio_anual['SALE DATE'].dt.year
  df_preciopromedio_anual=df_preciopromedio_anual.groupby(['County','SALE DATE']).mean().reset_index()
  df_preciopromedio_anual.columns=['Condado','Año','Precio por pie cuadrado Promedio']

  # Precio promedio de ventas de Condados

  df_precio_venta_promedio_anual=df_ventas_filtered[['County','SALE DATE','Precio_por_pie_cuadrado']].copy()
  df_precio_venta_promedio_anual['SALE DATE'] = df_precio_venta_promedio_anual['SALE DATE'].dt.year
  df_precio_venta_promedio_anual=df_precio_venta_promedio_anual.groupby(['County','SALE DATE']).mean().reset_index()
  df_precio_venta_promedio_anual.columns=['Condado','Año','Precio promedio']

  # Conteo de Ventas por Condado

  df_conteoventas_anual=df_ventas_filtered[['County','SALE DATE']].copy()
  df_conteoventas_anual['SALE DATE'] = df_conteoventas_anual['SALE DATE'].dt.year
  df_conteoventas_anual=df_conteoventas_anual.groupby(['County','SALE DATE']).value_counts().reset_index()
  df_conteoventas_anual.columns=['Condado','Año','Conteo']
  max_count=df_ventas_filtered['County'].value_counts().reset_index().sort_values(by='count',ascending=False).iloc[0,0]
  
  #Acomodo Historia NYC-------------------------------------------------------------------------------------------------------------------------------------------------------------------------


  cs11, cs12=st.columns((0.7,0.3))
  with cs11:
    st.subheader("Ventas en la ciudad de Nueva York")
    st.map=displaymapcount(df_ventas_filtered)
      
  with cs12:
    st.subheader("Métricas")
    container_stylesNumDuros('Condado con más ventas', max_count)
    container_stylesNumDuros('Número de Ventas', 
                             str('{:,}'.format(df_ventas_filtered['County'].value_counts().get(max_count, 0))))
    container_stylesNumDuros('Precio Promedio FT2', 
                             str('${:,.0f}'.format(df_ventas_filtered.groupby('County')['Precio_por_pie_cuadrado'].mean().get(max_count, 0))))
    container_stylesNumDuros('Suma de ventas', 
                             str('${:,.0f} M'.format(df_ventas_filtered.groupby('County')['SALE PRICE'].sum().get(max_count, 0)/1000000)))

  cs21, cs22=st.columns((2))
  with cs21:
    st.subheader("FT2 promedio de Condados")
    fig = px.line(df_preciopromedio_anual, x="Año", y="Precio por pie cuadrado Promedio",  color='Condado',
             color_discrete_map={
                 "Queens": "red",
                 "Bronx": "#17becf",
                 "Staten Island":"#1f77b4",
                 "Brooklyn":"#2ca02c",
                 "Manhattan":"#e377c2"
                 
             })
      
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))
    container_styles_plt(fig)

  with cs22:
    st.subheader("Conteo de Ventas por Condado")
    fig = px.line(df_conteoventas_anual, x="Año", y="Conteo", color='Condado',
             color_discrete_map={
                 "Queens": "red",
                 "Bronx": "#17becf",
                 "Staten Island":"#1f77b4",
                 "Brooklyn":"#2ca02c",
                 "Manhattan":"#e377c2"
                 
             })
      
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
        ))
    container_styles_plt(fig)

  st.subheader("Precio promedio de ventas por Condado")
  fig = px.line(df_precio_venta_promedio_anual, x="Año", y="Precio promedio",  color='Condado',
             color_discrete_map={
                 "Queens": "red",
                 "Bronx": "#17becf",
                 "Staten Island":"#1f77b4",
                 "Brooklyn":"#2ca02c",
                 "Manhattan":"#e377c2"
                 
             })
    
  fig.update_layout(legend=dict(
      orientation="h",
      yanchor="bottom",
      y=1.02,
      xanchor="right",
      x=1
  ))
  container_styles_plt(fig)


#Dashboard Queens-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

if selected == "Dashboard de Queens":
  st.title("Ventas en Queens")
   
  #Filtros opciones-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  st.sidebar.header("Filtros")

  Year=df_ventas['SALE DATE'].dt.year.sort_values(ascending=True).unique()
  building=df_ventas['BUILDING CLASS AT PRESENT'].sort_values(ascending=True).unique()   
  neighborhood=df_ventas['NEIGHBORHOOD'].sort_values(ascending=True).unique()   

  #Select Box Historia Queens----------------------------------------------------------------------------------------------------------------------------------------------------------------------

  # Select Box Año Historia
  year_option = st.sidebar.multiselect(
      "Año",
      [2019,2020,2021,2022,2023]
  )

  if not year_option:
    year_option=[2019,2020,2021,2022,2023]
   
  # Select Box Building Class
  building_option = st.sidebar.multiselect(
      "Building Class",
      ["A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "B1", "B2", "B3", "B9", "C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "CB", "CC", "CM"]
  )

  if (not building_option):
    building_option=["A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "B1", "B2", "B3", "B9", "C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "CB", "CC", "CM"]


  # Select Box Neighborhood
  neighborhood_option = st.sidebar.multiselect(
      "Neighborhood",
      ["SPRINGFIELD GARDENS", "SOUTH OZONE PARK", "HOLLIS", "CORONA", "GLENDALE", "EAST ELMHURST"]
  )

  if not neighborhood_option:
    neighborhood_option=["SPRINGFIELD GARDENS", "SOUTH OZONE PARK", "HOLLIS", "CORONA", "GLENDALE", "EAST ELMHURST"]
      
  #Notas--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  
  st.sidebar.write('Nota: Las Gráficas marcadas con * no tienen funcionalidad de filtros') 
    
  #Filtro en bases de datos Historia Queens--------------------------------------------------------------------------------------------------------------------------------------------------------------------------

  df_ventas_filtered=df_ventas[((df_ventas['BUILDING CLASS AT PRESENT'].isin(building_option)) & (df_ventas['SALE DATE'].dt.year.isin(year_option)) & (df_ventas['NEIGHBORHOOD'].isin(neighborhood_option)))]

  # Tipo de vivienda más vendida
  df_ventas_Btypes_sales=df_ventas.loc[(df_ventas['County']=='Queens')].copy()
  df_ventas_Btypes_sales['BUILDING CLASS AT PRESENT'] = df_ventas_Btypes_sales['BUILDING CLASS AT PRESENT'].str.slice(0, 1) + df_ventas_Btypes_sales['BUILDING CLASS AT PRESENT'].str.slice(2)
  df_ventas_Btypes_sales=df_ventas_Btypes_sales['BUILDING CLASS AT PRESENT'].value_counts().reset_index()
  df_ventas_Btypes_sales.columns=['Building Class Type','Conteo']
              # Top 7 Building Class Types
  top_building_types = df_ventas_Btypes_sales['Building Class Type'].head(5).tolist()

              # Replace Building Class Types not in the top 7 with 'Otros'
  df_ventas_Btypes_sales.loc[~df_ventas_Btypes_sales['Building Class Type'].isin(top_building_types), 'Building Class Type'] = 'Otros'
  df_ventas_Btypes_sales=df_ventas_Btypes_sales.groupby('Building Class Type')['Conteo'].sum().reset_index().sort_values(by='Conteo',ascending=False)

    
  # Consulta de más vendidos en Queens (Vecindad)

  df_conteo_neighborhood=df_ventas_filtered.loc[(df_ventas_filtered['County']=='Queens')]
  df_conteo_neighborhood=df_conteo_neighborhood['NEIGHBORHOOD'].value_counts().reset_index().sort_values(by='count',ascending=False)
  df_conteo_neighborhood.columns=['NEIGHBORHOOD','Conteo']
  df_conteo_neighborhood=df_conteo_neighborhood.head(10)

  # ft2 promedio por Vecindad Anual (Mejores vecindades)
  df_precio_venta_promedio_anual_queens=df_ventas_filtered[['NEIGHBORHOOD','SALE DATE','Precio_por_pie_cuadrado']].copy()
  df_precio_venta_promedio_anual_queens['SALE DATE'] = df_precio_venta_promedio_anual_queens['SALE DATE'].dt.year
  df_precio_venta_promedio_anual_queens=df_precio_venta_promedio_anual_queens.groupby(['NEIGHBORHOOD','SALE DATE']).mean().reset_index()
  df_precio_venta_promedio_anual_queens.columns=['Vecindario','Año','Precio promedio']

  #Ventas por Vecindario Anual (Mejores vecindades)

  df_conteoventas_anual_vecindario=df_ventas_filtered[['NEIGHBORHOOD','SALE DATE']].copy()
  df_conteoventas_anual_vecindario['SALE DATE'] = df_conteoventas_anual_vecindario['SALE DATE'].dt.year
  df_conteoventas_anual_vecindario=df_conteoventas_anual_vecindario.groupby(['SALE DATE']).value_counts().reset_index()
  df_conteoventas_anual_vecindario.columns=['Año','Vecindario','Conteo']

  # Consulta de más vendidos en Queens (tipo de Propiedad)

  df_conteo_building=df_ventas_filtered.loc[(df_ventas_filtered['County']=='Queens')]
  df_conteo_building=df_conteo_building['BUILDING CLASS AT PRESENT'].value_counts().reset_index().sort_values(by='count',ascending=False)
  df_conteo_building.columns=['BUILDING CLASS AT PRESENT','Conteo']
  df_conteo_building=df_conteo_building.head(10)

  #Ventas Totales (Sale price)
  df_conteo_ventas_total_queens=df_ventas.loc[df_ventas['County']=='Queens']
  df_queens_2023=df_conteo_ventas_total_queens.loc[(df_conteo_ventas_total_queens['SALE DATE'].dt.year == 2023) & (df_ventas['BUILDING CLASS AT PRESENT'].str.contains('A') | df_ventas['BUILDING CLASS AT PRESENT'].str.contains('B') | df_ventas['BUILDING CLASS AT PRESENT'].str.contains('C'))&(df_ventas['NEIGHBORHOOD'].isin(["SPRINGFIELD GARDENS", "SOUTH OZONE PARK", "HOLLIS", "CORONA", "GLENDALE", "EAST ELMHURST"]))]
  df_queens_2020=df_conteo_ventas_total_queens.loc[(df_conteo_ventas_total_queens['SALE DATE'].dt.year == 2020) & (df_ventas['BUILDING CLASS AT PRESENT'].str.contains('A') | df_ventas['BUILDING CLASS AT PRESENT'].str.contains('B') | df_ventas['BUILDING CLASS AT PRESENT'].str.contains('C'))&(df_ventas['NEIGHBORHOOD'].isin(["SPRINGFIELD GARDENS", "SOUTH OZONE PARK", "HOLLIS", "CORONA", "GLENDALE", "EAST ELMHURST"]))]


  #Acomodo Historia Queens-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  cs11, cs12, cs13=st.columns((3))
  with cs11:
    container_stylesNumDuros('Precio de Venta total', 
                             str('${:,.0f} M'.format(df_ventas_filtered.groupby('County')['SALE PRICE'].sum().get('Queens', 0)/1000000)))

  with cs12:
    container_stylesNumDuros('Precio Promedio FT2',
                             str('${:,.0f}'.format(df_ventas_filtered.groupby('County')['Precio_por_pie_cuadrado'].mean().get('Queens', 0))))

  with cs13:
    container_stylesNumDuros('Numero de ventas', 
                             str('{:,}'.format(df_ventas_filtered['County'].value_counts().get('Queens', 0))))

  cs21, cs22=st.columns((0.4,0.6)) 
  with cs21:
      st.subheader("* Edificios con más ventas")
      fig = px.pie(df_ventas_Btypes_sales,values="Conteo", names="Building Class Type")
      fig.update_layout(legend=dict(
          orientation="h",
          yanchor="bottom",
          y=1.02,
          xanchor="right",
          x=1
      ))  
      container_styles_plt(fig)


  with cs22: 
      st.subheader("Ventas por Tipo de edificio")
      fig = px.bar(df_conteo_building, x="BUILDING CLASS AT PRESENT", y="Conteo",
          text= ['{:,}'.format(x) for x in df_conteo_building["Conteo"]], template="seaborn", color_discrete_sequence=["#2D2E78"])
      container_styles_plt(fig)

  cs31, cs32=st.columns((2))
  with cs31:
    st.subheader("Ventas por vecindario Anual")
    fig = px.line(df_conteoventas_anual_vecindario, x="Año", y="Conteo", color='Vecindario',
             color_discrete_map={
                 "SPRINGFIELD GARDENS": "red",
                 "SOUTH OZONE PARK": "#17becf",
                 "HOLLIS":"#1f77b4",
                 "CORONA":"#2ca02c",
                 "GLENDALE":"#e377c2",
                 "EAST ELMHURST":"#8c564b"
                 
             })
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))
    container_styles_plt(fig)

  with cs32:
    st.subheader("Precio por FT2 promedio Anual")
    fig = px.line(df_precio_venta_promedio_anual_queens, x="Año", y="Precio promedio", color='Vecindario',
             color_discrete_map={
                 "SPRINGFIELD GARDENS": "red",
                 "SOUTH OZONE PARK": "#17becf",
                 "HOLLIS":"#1f77b4",
                 "CORONA":"#2ca02c",
                 "GLENDALE":"#e377c2",
                 "EAST ELMHURST":"#8c564b"
                 
             })   
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))
    container_styles_plt(fig)
      
  st.subheader("* Estadísticas Generales de Queens 2020")
  cs41, cs42, cs43=st.columns((3))
  with cs41:
    with stylable_container(

          key="styContnum",
          css_styles="""
              {
                  background-color: #FFFFFF;
                  border: 1px solid #CCCCCC;
                  padding: 5% 5% 5% 10%;
                  border-radius: 5px;

                  border-left: 0.5rem solid #9AD8E1 !important;
                  box-shadow: 0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.15) !important;
              }
              """,
      ):
          st.metric('Número de ventas',
                    str('{:,.0f}'.format(df_queens_2020.shape[0])))

  with cs42:
    with stylable_container(

          key="styContnum",
          css_styles="""
              {
                  background-color: #FFFFFF;
                  border: 1px solid #CCCCCC;
                  padding: 5% 5% 5% 10%;
                  border-radius: 5px;

                  border-left: 0.5rem solid #9AD8E1 !important;
                  box-shadow: 0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.15) !important;
              }
              """,
      ):
          st.metric('Valor del mercado',
                    str('${:,.0f} M'.format(df_queens_2020['SALE PRICE'].sum()/1000000)))

  with cs43:
    with stylable_container(

          key="styContnum",
          css_styles="""
              {
                  background-color: #FFFFFF;
                  border: 1px solid #CCCCCC;
                  padding: 5% 5% 5% 10%;
                  border-radius: 5px;

                  border-left: 0.5rem solid #9AD8E1 !important;
                  box-shadow: 0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.15) !important;
              }
              """,
      ):
          st.metric('Precio por FT2 Promedio',
                    str('${:,.0f}'.format(df_queens_2020['Precio_por_pie_cuadrado'].mean())))

  st.subheader("* Estadísticas Generales de Queens 2023")  
  cs51, cs52, cs53=st.columns((3))
  with cs51:
    with stylable_container(

          key="styContnum",
          css_styles="""
              {
                  background-color: #FFFFFF;
                  border: 1px solid #CCCCCC;
                  padding: 5% 5% 5% 10%;
                  border-radius: 5px;

                  border-left: 0.5rem solid #9AD8E1 !important;
                  box-shadow: 0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.15) !important;
              }
              """,
      ):
          st.metric('Número de ventas',
                    str('{:,.0f}'.format(df_queens_2023.shape[0])),
                    str('{:,.0f}%'.format((df_queens_2023.shape[0]/df_queens_2020.shape[0])*100-100))
                   )

  with cs52:
    with stylable_container(

          key="styContnum",
          css_styles="""
              {
                  background-color: #FFFFFF;
                  border: 1px solid #CCCCCC;
                  padding: 5% 5% 5% 10%;
                  border-radius: 5px;

                  border-left: 0.5rem solid #9AD8E1 !important;
                  box-shadow: 0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.15) !important;
              }
              """,
      ):
          st.metric('Valor del mercado',
                    str('${:,.0f} M'.format(df_queens_2023['SALE PRICE'].sum()/1000000)),
                    str('{:,.0f}%'.format((df_queens_2023['SALE PRICE'].sum()/df_queens_2020['SALE PRICE'].sum())*100-100))
                   )

  with cs53:
    with stylable_container(

          key="styContnum",
          css_styles="""
              {
                  background-color: #FFFFFF;
                  border: 1px solid #CCCCCC;
                  padding: 5% 5% 5% 10%;
                  border-radius: 5px;

                  border-left: 0.5rem solid #9AD8E1 !important;
                  box-shadow: 0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.15) !important;
              }
              """,
      ):
          st.metric('Precio por FT2 Promedio',
                    str('${:,.0f}'.format(df_queens_2023['Precio_por_pie_cuadrado'].mean())),
                    str('{:,.0f}%'.format((df_queens_2023['Precio_por_pie_cuadrado'].mean()/df_queens_2020['Precio_por_pie_cuadrado'].mean())*100-100))
                   )


if selected == "Pronóstico de Queens":

  # Selección de periodo--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  st.title('Pronóstico del Precio por FT2')
  st.write('En este apartado se pronostica el precio por FT2 en los vecindarios de **Springfield Gardens, South Ozone Park, Hollis, Corona, Glendale, East Elmhurst**; para las propiedades **A, B y C**. A un año (**Finales del 2024-Inicios del 2025**).')
    
  st.sidebar.title('')
  frec_option = st.sidebar.selectbox(
      "Frecuencia de proyección",
      ['Diario', 'Semanal','Quincenal','Mensual','Trimestre']
  )

  if frec_option=='Mensual':
    frec='M'
    period = st.sidebar.slider("Proyección de (Mes): ", 1, 13, 13)  

  elif frec_option=='Diario':
    frec='D'
    period = st.sidebar.slider("Proyección de (Diario): ", 100, 395, 395)

  elif frec_option=='Trimestre':
    frec='Q'
    period = st.sidebar.slider("Proyección de (Trimestre): ", 1, 4, 4) 

  elif frec_option=='Semanal':
    frec='W'
    period = st.sidebar.slider("Proyección de (Semana): ", 1, 56, 56)
      
  elif frec_option=='Quincenal':
    frec='SM'
    period = st.sidebar.slider("Proyección de (Semana): ", 1, 28, 28)

   
  #Check Box Bajo rendimiento-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  
  alto_rendimiento = st.sidebar.checkbox("Seleccione aquí para obtener una gráfica de proyección con detalles")
    
  st.sidebar.write('Nota: La gráfica a detalle puede consumir demasiados recursos')
  
  #st.sidebar.write("Nota: Solo funciona con gráficas oscuras")
     
  #Filtro en bases de datos Pronóstico--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  df_ventas_queens=df_queens(df_ventas,frec)
  m=prophet(df_ventas_queens)
  forecast=forecast(period,frec, m)
  df_conteo_ventas_total_queens_prom=df_ventas.loc[(df_ventas['County']=='Queens')&(df_ventas['SALE DATE'].dt.year.isin([2019,2020,2021,2022,2023]))&(df_ventas['BUILDING CLASS AT PRESENT'].str.contains('A') | df_ventas['BUILDING CLASS AT PRESENT'].str.contains('B') | df_ventas['BUILDING CLASS AT PRESENT'].str.contains('C'))&(df_ventas['NEIGHBORHOOD'].isin(["SPRINGFIELD GARDENS", "SOUTH OZONE PARK", "HOLLIS", "CORONA", "GLENDALE", "EAST ELMHURST"]))]['Precio_por_pie_cuadrado'].mean()
  forecast2023=forecast.loc[forecast['ds'].dt.year==2023].sort_values('ds').tail(1)
  forecast2024=forecast.loc[forecast['ds'].dt.year==2024].sort_values('ds').tail(1)
  
  #Acomodo Pronóstico-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

  st.subheader("Proyección a futuro")

  if alto_rendimiento:
      fig = plot_plotly(m, forecast)
      fig.update_layout(paper_bgcolor='rgba(0, 0, 0, 0)')
      fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)')
      fig.update_layout(
              autosize=True,
              margin=dict(
                  l=50,
                  r=50,
                  b=100,
                  t=100,
                  pad=4
              )
          )

      with stylable_container(
        
          key="styContplt",
          css_styles="""
              {
                  background-color: #FFFFFF;
                  border: 1px solid #CCCCCC;
                  padding: 0% 5% 3% 0%;
                  border-radius: 5px;
        
                  border-left: 0.5rem solid #9AD8E1 !important;
                  box-shadow: 0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.15) !important;
              }
              """,
      ): st.plotly_chart(fig)
    
  else:
      fig = m.plot(forecast)
      st.session_state.figr=fig
      with stylable_container(
        
          key="styContplt",
          css_styles="""
              {
                  background-color: #FFFFFF;
                  border: 1px solid #CCCCCC;
                  padding: 10% 5% 3% 3%;
                  border-radius: 5px;
        
                  border-left: 0.5rem solid #9AD8E1 !important;
                  box-shadow: 0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.15) !important;
              }
              """,
      ):cs66,cs77=st.columns((0.98,0.02))
      with cs66: st.write(fig)
      with cs77: st.write('')


  st.subheader("Estadísticas de Proyección")
  cs51,cs52,cs53=st.columns((3))
  with cs51: 
    with stylable_container(

          key="styContnum",
          css_styles="""
              {
                  background-color: #FFFFFF;
                  border: 1px solid #CCCCCC;
                  padding: 5% 5% 5% 10%;
                  border-radius: 5px;

                  border-left: 0.5rem solid #9AD8E1 !important;
                  box-shadow: 0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.15) !important;
              }
              """,
      ):
          st.metric('Máximo',
                    str('${:,.2f}'.format(forecast2024.loc[forecast2024.index[0],'yhat_upper'])),
                    str('{:,.0f}%'.format((forecast2024.loc[forecast2024.index[0],'yhat_upper']/df_conteo_ventas_total_queens_prom)*100-100))
                   )
  with cs52: 
    with stylable_container(

          key="styContnum",
          css_styles="""
              {
                  background-color: #FFFFFF;
                  border: 1px solid #CCCCCC;
                  padding: 5% 5% 5% 10%;
                  border-radius: 5px;

                  border-left: 0.5rem solid #9AD8E1 !important;
                  box-shadow: 0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.15) !important;
              }
              """,
      ):
          st.metric('Neutro',
                    str('${:,.2f}'.format(forecast2024.loc[forecast2024.index[0],'trend'])),
                    str('{:,.0f}%'.format((forecast2024.loc[forecast2024.index[0],'trend']/df_conteo_ventas_total_queens_prom)*100-100))
                   )
  with cs53:
    with stylable_container(

          key="styContnum",
          css_styles="""
              {
                  background-color: #FFFFFF;
                  border: 1px solid #CCCCCC;
                  padding: 5% 5% 5% 10%;
                  border-radius: 5px;

                  border-left: 0.5rem solid #9AD8E1 !important;
                  box-shadow: 0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.15) !important;
              }
              """,
      ):
          st.metric('Mínimo',
                    str('${:,.2f}'.format(forecast2024.loc[forecast2024.index[0],'yhat_lower'])),
                    str('{:,.0f}%'.format((forecast2024.loc[forecast2024.index[0],'yhat_lower']/df_conteo_ventas_total_queens_prom)*100-100))
                   )

    
  st.subheader("Tendencia")
  fig = go.Figure()
  fig.add_trace(go.Scatter(x=forecast['ds'].tolist(), y=forecast['trend'].tolist(),
                      mode='lines',
                      name='Tendencia'))
  fig.add_trace(go.Scatter(x=forecast['ds'].tolist(), y=forecast['yhat_upper'].tolist(),
                      mode='lines',
                      name='Máximo'))
  fig.add_trace(go.Scatter(x=forecast['ds'].tolist(), y=forecast['yhat_lower'].tolist(),
                      mode='lines', name='Mínimo'))
  
  fig.update_layout(legend=dict(
      orientation="h",
      yanchor="bottom",
      y=1.02,
      xanchor="right",
      x=1
  ))
  container_styles_plt(fig)



  st.subheader("Datos")
  st.dataframe(forecast[['ds','trend','yhat_lower','yhat_upper']],use_container_width=True)


if selected == "Dashboard de usuario":

  st.title("Ventas en NYC")  
   
  #Select Box Condado (De prioridad)-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  county_option = st.sidebar.multiselect(
      "Condado",
      df_ventas['County'].unique()
  )

  if not county_option:
    county_option=df_ventas['County'].unique()
      
  df_ventas_filtered=df_ventas[df_ventas['County'].isin(county_option)]
 
  #Filtros opciones-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  
  st.sidebar.header("Filtros")

  Year=df_ventas_filtered['SALE DATE'].dt.year.sort_values(ascending=True).unique()
  building=df_ventas_filtered['BUILDING CLASS AT PRESENT'].sort_values(ascending=True).unique()     
  neighborhood=df_ventas_filtered['NEIGHBORHOOD'].sort_values(ascending=True).unique()   

  #Select Box----------------------------------------------------------------------------------------------------------------------------------------------------------------------

  # Select Box Año
  year_option = st.sidebar.multiselect(
      "Año",
      Year
  )

  if not year_option:
    year_option=df_ventas_filtered['SALE DATE'].dt.year.unique()
   
  # Select Box Building Class
  building_option = st.sidebar.multiselect(
      "Building Class",
      building
  )

  if (not building_option):
    building_option=df_ventas_filtered['BUILDING CLASS AT PRESENT'].unique()   


  # Select Box Neighborhood
  neighborhood_option = st.sidebar.multiselect(
      "Neighborhood",
      neighborhood
  )

  if not neighborhood_option:
    neighborhood_option=df_ventas_filtered['NEIGHBORHOOD'].unique()   
      
  #Notas--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  
  st.sidebar.write('Nota: Las Gráficas marcadas con * no tienen funcionalidad de filtros') 
    
  #Filtro en bases de datos Historia --------------------------------------------------------------------------------------------------------------------------------------------------------------------------

  df_ventas_filtered=df_ventas[((df_ventas['BUILDING CLASS AT PRESENT'].isin(building_option)) & (df_ventas['SALE DATE'].dt.year.isin(year_option)) & (df_ventas['NEIGHBORHOOD'].isin(neighborhood_option)))]

  # Tipo de vivienda más vendida

  df_ventas_Btypes_sales=df_ventas_filtered.copy()
  df_ventas_Btypes_sales['BUILDING CLASS AT PRESENT'] = df_ventas_Btypes_sales['BUILDING CLASS AT PRESENT'].str.slice(0, 1) + df_ventas_Btypes_sales['BUILDING CLASS AT PRESENT'].str.slice(2)
  df_ventas_Btypes_sales=df_ventas_Btypes_sales['BUILDING CLASS AT PRESENT'].value_counts().reset_index()
  df_ventas_Btypes_sales.columns=['Building Class Type','Conteo']
          # Top 7 Building Class Types
  top_building_types = df_ventas_Btypes_sales['Building Class Type'].head(5).tolist()

          # Replace Building Class Types not in the top 7 with 'Otros'
  df_ventas_Btypes_sales.loc[~df_ventas_Btypes_sales['Building Class Type'].isin(top_building_types), 'Building Class Type'] = 'Otros'
  df_ventas_Btypes_sales=df_ventas_Btypes_sales.groupby('Building Class Type')['Conteo'].sum().reset_index().sort_values(by='Conteo',ascending=False)

  # Consulta de más vendidos (Vecindad)

  df_conteo_neighborhood=df_ventas_filtered.copy()
  df_conteo_neighborhood=df_conteo_neighborhood['NEIGHBORHOOD'].value_counts().reset_index().sort_values(by='count',ascending=False)
  df_conteo_neighborhood.columns=['NEIGHBORHOOD','Conteo']
  df_conteo_neighborhood=df_conteo_neighborhood.head(10)

  # ft2 promedio por Vecindad Anual

          #df_precio_venta_promedio_anual=df_ventas_filtered[['NEIGHBORHOOD','SALE DATE','Precio_por_pie_cuadrado']].copy()
          #df_precio_venta_promedio_anual['SALE DATE'] = df_precio_venta_promedio_anual['SALE DATE'].dt.year
          #df_precio_venta_promedio_anual=df_precio_venta_promedio_anual.groupby(['NEIGHBORHOOD','SALE DATE']).mean().reset_index()
          #df_precio_venta_promedio_anual.columns=['Vecindario','Año','Precio promedio']
          #df_precio_venta_promedio_anual=df_precio_venta_promedio_anual.head(7)
    
  df_precio_venta_promedio_anual=df_ventas_filtered[['NEIGHBORHOOD','SALE DATE','Precio_por_pie_cuadrado']].copy()
  df_precio_venta_promedio_anual['SALE DATE'] = df_precio_venta_promedio_anual['SALE DATE'].dt.year
  df_precio_venta_promedio_anual=df_precio_venta_promedio_anual.groupby(['NEIGHBORHOOD','SALE DATE']).mean().reset_index()
  df_precio_venta_promedio_anual.columns=['Vecindario','Año','Precio promedio']
              # Top 7 neighborhoods
  top_neighborhoods_feet = df_ventas_filtered[['NEIGHBORHOOD','SALE DATE','Precio_por_pie_cuadrado']].groupby(['NEIGHBORHOOD','SALE DATE']).mean().reset_index().sort_values(by='Precio_por_pie_cuadrado',ascending=False)['NEIGHBORHOOD'].head(7).tolist()
              # Filtro
  df_precio_venta_promedio_anual = df_precio_venta_promedio_anual[df_precio_venta_promedio_anual['Vecindario'].isin(top_neighborhoods_feet)]

  #Ventas por Vecindario Anual

          #df_conteoventas_anual_vecindario=df_ventas_filtered[['NEIGHBORHOOD','SALE DATE']].copy()
          #df_conteoventas_anual_vecindario['SALE DATE'] = df_conteoventas_anual_vecindario['SALE DATE'].dt.year
          #df_conteoventas_anual_vecindario=df_conteoventas_anual_vecindario.groupby(['SALE DATE']).value_counts().reset_index()
          #df_conteoventas_anual_vecindario.columns=['Año','Vecindario','Conteo']
          #df_conteoventas_anual_vecindario=df_conteoventas_anual_vecindario.head(7)

  df_conteoventas_anual_vecindario=df_ventas_filtered[['NEIGHBORHOOD','SALE DATE']].copy()
  df_conteoventas_anual_vecindario['SALE DATE'] = df_conteoventas_anual_vecindario['SALE DATE'].dt.year
  df_conteoventas_anual_vecindario=df_conteoventas_anual_vecindario.groupby(['SALE DATE']).value_counts().reset_index()
  df_conteoventas_anual_vecindario.columns=['Año','Vecindario','Conteo']
              # Top 7 neighborhoods
  top_neighborhoods = df_ventas_filtered['NEIGHBORHOOD'].value_counts().reset_index().sort_values(by='count',ascending=False)['NEIGHBORHOOD'].head(7).tolist()
              # Filtro
  df_conteoventas_anual_vecindario = df_conteoventas_anual_vecindario[df_conteoventas_anual_vecindario['Vecindario'].isin(top_neighborhoods)]

  # Consulta de más vendidos (tipo de Propiedad)

  df_conteo_building=df_ventas_filtered.loc[(df_ventas_filtered['County']=='Queens')]
  df_conteo_building=df_conteo_building['BUILDING CLASS AT PRESENT'].value_counts().reset_index().sort_values(by='count',ascending=False)
  df_conteo_building.columns=['BUILDING CLASS AT PRESENT','Conteo']
  df_conteo_building=df_conteo_building.head(10)


  #Acomodo Historia-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  cs11, cs12, cs13=st.columns((3))
  with cs11:
    container_stylesNumDuros('Precio de venta total', str('${:,.0f} M'.format(df_ventas_filtered['SALE PRICE'].sum()/1000000)))

  with cs12:
    container_stylesNumDuros('Precio Promedio FT2',str('${:,.0f}'.format(df_ventas_filtered['Precio_por_pie_cuadrado'].mean())))

  with cs13:
    container_stylesNumDuros('Numero de ventas', str('{:,}'.format(df_ventas_filtered.shape[0])))

    
  cs21, cs22=st.columns((0.4,0.6)) 
  with cs21:
      st.subheader("Edificios con más ventas")
      fig = px.pie(df_ventas_Btypes_sales, values="Conteo", names="Building Class Type")
      fig.update_layout(legend=dict(
          orientation="h",
          yanchor="bottom",
          y=1.02,
          xanchor="right",
          x=1
      ))
      container_styles_plt(fig)
      
  with cs22:
      st.subheader("Ventas por Tipo de edificio")
      fig = px.bar(df_conteo_building, x="BUILDING CLASS AT PRESENT", y="Conteo",
          text= ['{:,}'.format(x) for x in df_conteo_building["Conteo"]], template="seaborn", color_discrete_sequence=["#2D2E78"])
      container_styles_plt(fig)

  cs31, cs32=st.columns((2))
  with cs31:
    st.subheader("Ventas por vecindario Anual")
    fig = px.line(df_conteoventas_anual_vecindario, x="Año", y="Conteo", color='Vecindario')
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))
    container_styles_plt(fig)

  with cs32:
    st.subheader("Precio por FT2 promedio Anual")
    fig = px.line(df_precio_venta_promedio_anual, x="Año", y="Precio promedio", color='Vecindario')   
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))
    container_styles_plt(fig)
 

if selected == "Base de Datos":
  cs11, cs12, cs13=st.columns((0.2,0.6,0.2))  
  with cs11:
    st.write('')

  with cs12:    
    st.download_button(
        use_container_width=True,
        label="Descargar Base de Datos como CSV",
        data=df_ventas.to_csv(),
        file_name="NYC_Sales.csv",
        mime="text/csv",
    )
      
  with cs13:
    st.write('')
