"""
# My first app
Here's our first attempt at using data to create a table:
"""
import plotly.figure_factory as ff
import streamlit as st
import numpy as np
import math
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, least_squares
import seaborn as sns
from datetime import datetime
sns.set_style('whitegrid')

monitorings = pd.read_csv('monitoring_cleaned.csv')
print("columns below")
print(monitorings.columns)

monitorings['FechaMuestreo'] = pd.to_datetime(monitorings['FechaMuestreo'])

monitorings['feed_percent_biomass'] =   (monitorings['KilosAlimento']/7) / monitorings['live_biomass']
print(monitorings.columns)
max_date = monitorings['FechaMuestreo'].max().date()
min_date = monitorings['FechaMuestreo'].min().date()

harvests = pd.read_csv('bravito_harvests.csv')

#last_cycle = monitorings.loc[monitorings.groupby('')]

def get_decreasing(df):
    data = (
        df.groupby([
            'PKCiclo', pd.Grouper(
                key='FechaMuestreo', freq='1W',
            ),
        ])
        .PesoPromedio2.median()
        .reset_index()
    )

    window_diff = (
        data.groupby('PKCiclo')
        .rolling(window=2)
        .PesoPromedio2.apply(np.diff)
        .dropna()
        .reset_index()
        .drop(columns=['level_1'])
    )
    return window_diff

def clip_extreme_change(df, mini, maxi):
    d = get_decreasing(df)
    normal_population = d[
        (d.PesoPromedio2 < mini) | (d.PesoPromedio2 > maxi)
    ].PKCiclo.unique()
    return df.loc[~df.PKCiclo.isin(normal_population)].copy()

def dropna(df, subset=['PesoPromedio2', 'cycle_days']):
  return df.dropna(subset = subset)

def clip_extreme_growth(df, mini, maxi):
  df['daily_growth'] = df['PesoPromedio2'] / df['cycle_days']
  growth_median = df['daily_growth'].median()
  mini_growth = growth_median * mini
  maxi_growth = growth_median * maxi
  return df[(df['daily_growth']>=mini_growth) & (df['daily_growth']<=maxi_growth)]

def print_shape(df):
  print(df.shape)
  return df


clean_df = (monitorings.pipe(print_shape)
             .pipe(clip_extreme_change,-1, 6)
             .pipe(print_shape)
             .pipe(clip_extreme_growth,0.3,1.75)

             .pipe(print_shape)
             .dropna()
             )

def growth_model(time, Linf, K, position, w0):
    return Linf * np.exp(-np.exp(position - K * time)) + w0


def get_benchmark(df, weight_min, weight_max, model):
  train_df = df[(df['PesoPromedio2'] > weight_min) & (df['PesoPromedio2'] < weight_max)]
  x_train = df['cycle_days'] 
  y_train = df['PesoPromedio'] 

  curve_params, covariance = curve_fit(model,
                                     x_train,
                                     y_train,
                                     p0 = [37.728, 0.02348,0.93, 8.008],
                                     maxfev = 100000
                                          )
  return curve_params


#dictionaries -------------------------------------------------------------
param_dict = {
    'Supervivencia': [0.7, 1.5, 50,100],
    'biomass_ha': [0.5, 2, 50,10000],
    'PesoPromedio2':[0.5, 1.5, 2,35],
    'cumulative_fcr':[0.25, 1.5, 0.1,3],
    'weekly_fcr':[0.1, 2, 0.1,5],
    '1week_growth_rate':[0.5, 1.5, 0.1,5],
    '2week_growth_rate':[0.5, 2, 0.1,5],
    'kg/ha/day':[0.01, 4, 0.1,120],
    'feed_percent_biomass':[0.01, 10, 0,2000],
    'mlResultWeightCv':[0.01, 10, 0,2000]
}

labels_dict = {
    'Supervivencia': 'Survival Rate',
    'biomass_ha': 'Biomass/Ha',
    'PesoPromedio2':'Average Weight',
    'cumulative_fcr':'Cumulative FCR',
    'weekly_fcr':'Weekly FCR',
    '1week_growth_rate':'Growth Rate - 1 Week',
    '2week_growth_rate':'Growth Rate - 2 Week',
    'kg/ha/day': 'KG/Ha/Day',
    'feed_percent_biomass': "Feed - % of biomass",
    'mlResultWeightCv':"CV"
}
labels_reverse_dict = dict((v,k) for k,v in labels_dict.items())
active_cycles = pd.read_csv('active_cycles.csv')

active_cycles.sort_values('pondName', inplace = True)
pond_cycle_dict = active_cycles.set_index('pondName')['PKCiclo'].to_dict()





def clean_df(df, x_variable, y_variable, min_threshold, max_threshold, absolute_min, absolute_max, start_date,end_date):
  print(df.shape)
  print(df.columns)
  df = df.dropna(subset=[x_variable, y_variable]).copy()
  df = df[(df[y_variable] < absolute_max) 
          & (df[y_variable] > absolute_min) 
          & (df['cycle_days'] <90)
          & (df['FechaMuestreo'].dt.date >= start_date)
          & (df['FechaMuestreo'].dt.date <= end_date)
          ]

  if y_variable == 'Supervivencia':
    df['rate'] = (1 - (df[y_variable]/100)) / df[x_variable]

  elif y_variable in ['1week_growth_rate', '2week_growth_rate']:
    df['rate'] = df[y_variable]

  else:
    df['rate'] = df[y_variable] / df[x_variable]
  
  median = df['rate'].median()
  print(median)
  min_rate = median *  min_threshold
  max_rate = median *  max_threshold

  filtered_df = df[(df['rate'] >= min_rate) & (df['rate'] <= max_rate)]
  print(filtered_df.shape)
  return filtered_df



def get_curve_params(df,y_variable, model, x_variable = 'cycle_days'):
  x_train = df[x_variable]
  print(x_train.iloc[:5])
  print(x_train.shape)
  y_train = df[y_variable]
  print(y_variable)

  curve_params, covariance = curve_fit(model,
                                     x_train,
                                     y_train,
                                     maxfev = 100000
                                          )
  return curve_params 


def plot_benchmark(curve_params, model, x_min, x_max, increment, x_label, y_label):
  x_plot = np.arange(x_min, x_max, increment)

  y_plot = np.round(model(x_plot, *curve_params),3)

  return pd.DataFrame({x_label:x_plot,
                       y_label:y_plot
                       })

#models ---------------------------------------------------
def growth_model(time, Linf, K, position, w0):
    return Linf * np.exp(-np.exp(position - K * time)) + w0


def exponential_fit(x, a,b,c,d):
  return a*x**3 + b*x**2 + c*x + d

def exponential_fit_2_degrees(x, b,c,d):
  return b*x**2 + c*x + d
def exponential_decay(N0, k, t):
    return N0 * math.exp(-k * t)
#sidebar ---------------------------------------------------
cycle_options = pond_cycle_dict.keys() 





sidebar_var1 = st.sidebar.selectbox(
    "Metric #1",
    list(labels_reverse_dict.keys()),
 
    placeholder="Metric #1",
    )

sidebar_var2 = st.sidebar.selectbox(
    "Metric #2",
    list(labels_reverse_dict.keys()),

    placeholder="Metric #2",
    )


sidebar_cycle = st.sidebar.selectbox(
    "Cycle",
    cycle_options,
    index=None,
    placeholder="Select Cycle",
    )

start_time, end_time = st.sidebar.slider(
        "Benchmark Window",
        value=[min_date,max_date],
        format="MM/DD/YY")

show_benchmarks = st.sidebar.toggle('Show Benchmarks', value = True)

second_graph = st.sidebar.toggle('Show Second Graph')
show_raleos = st.sidebar.toggle('Show Raleos', value = False)

if second_graph:
    sidebar_var3 = st.sidebar.selectbox(
        "Metric #3",
        list(labels_reverse_dict.keys()),
        placeholder="Metric #3",
        )

    sidebar_var4 = st.sidebar.selectbox(
        "Metric #4",
        list(labels_reverse_dict.keys()),
        placeholder="Metric #4",
        )

    y_variable3 = labels_reverse_dict[sidebar_var3]
    y_variable4 = labels_reverse_dict[sidebar_var4]
    


   

   
y_variable1 = labels_reverse_dict[sidebar_var1]
y_variable2 = labels_reverse_dict[sidebar_var2]
cycle_id = pond_cycle_dict[sidebar_cycle]
        






def get_variable_df(df, y_variable, model,start_time, end_time):
    cleaned_df = clean_df(df, 'cycle_days', y_variable, *param_dict[y_variable],start_time, end_time)

    if y_variable == 'mlResultWeightCv':
       curve_params = get_curve_params(cleaned_df[cleaned_df['cycle_days']>15], y_variable,exponential_fit_2_degrees)
       plot_df = plot_benchmark(curve_params,exponential_fit_2_degrees, 5, 91, 1, 'Cycle_Day', y_variable)
    else:
       curve_params = get_curve_params(cleaned_df, y_variable,model)
       plot_df = plot_benchmark(curve_params,model, 5, 91, 1, 'Cycle_Day', y_variable)
    print(curve_params)
    
    if y_variable == 'Supervivencia':
       plot_df.to_csv('bravito_survival_benchmark.csv')

    return plot_df 



variable1_df = get_variable_df(monitorings, y_variable1, exponential_fit, start_time, end_time)
variable2_df = get_variable_df(monitorings, y_variable2, exponential_fit, start_time, end_time)

if second_graph:
    variable3_df = get_variable_df(monitorings, y_variable3, exponential_fit, start_time, end_time)
    variable4_df = get_variable_df(monitorings, y_variable4, exponential_fit, start_time, end_time)
        
    plot_current_cycle3 = monitorings.loc[monitorings['PKCiclo'] == cycle_id, ['cycle_days', y_variable3]]


    plot_current_cycle4 = monitorings.loc[monitorings['PKCiclo'] == cycle_id, ['cycle_days', y_variable4]]





plot_current_cycle = monitorings.loc[monitorings['PKCiclo'] == cycle_id, ['cycle_days', y_variable1]]


plot_current_cycle2 = monitorings.loc[monitorings['PKCiclo'] == cycle_id, ['cycle_days', y_variable2]]

    
print(plot_current_cycle)
print(harvests.dtypes)
cycle_raleos =  harvests.loc[
                        (harvests['Parcial'] == 1) & 
                        (harvests['PKCiclo'] == cycle_id)]
print(harvests)
print(cycle_raleos)

import plotly.graph_objects as go
from plotly.subplots import make_subplots


    # Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig2 = make_subplots(specs=[[{"secondary_y": True}]])

layout = go.Layout(
  margin=go.layout.Margin(
        l=0, #left margin
        r=0, #right margin
        b=0, #bottom margin
        t=0, #top margin
    ))
    # Add traces
if show_benchmarks:
        fig.add_trace(
            go.Scatter(x=variable1_df['Cycle_Day'], 
                    y=variable1_df[y_variable1], 
                    name=labels_dict[y_variable1],
                    line=dict(color="#0068c9", dash="dash")),
            secondary_y=False,
            
        )

fig.add_trace(
        go.Scatter(x=plot_current_cycle['cycle_days'], 
                y=plot_current_cycle[y_variable1], 
                name= "Current Cycle " + labels_dict[y_variable1],
                line=dict(color="#0068c9")
                
                
                ),
        secondary_y=False,
        
    )

   

if show_benchmarks:
    fig.add_trace(
            go.Scatter(x=variable2_df['Cycle_Day'], 
                    y=variable2_df[y_variable2], 
                    name=labels_dict[y_variable2],
                    line=dict(color="#83c9ff",dash="dash")
                    
                    ),
            secondary_y=True,
            
        )

fig.add_trace(
    go.Scatter(x=plot_current_cycle2['cycle_days'], 
                y=plot_current_cycle2[y_variable2], 
                name= "Current Cycle " + labels_dict[y_variable2],
                line=dict(color="#83c9ff")
                
                
                ),
        secondary_y=True,
        
    )

if second_graph:
    if show_benchmarks:
        fig2.add_trace(
                    go.Scatter(x=variable3_df['Cycle_Day'], 
                            y=variable3_df[y_variable3], 
                            name=labels_dict[y_variable3],
                            line=dict(color="#FFB983", dash="dash")),
                    secondary_y=False,
                    
                )  
        fig2.add_trace(
                    go.Scatter(x=variable4_df['Cycle_Day'], 
                            y=variable4_df[y_variable4], 
                            name=labels_dict[y_variable4],
                            line=dict(color="#C900BB", dash="dash")),
                    secondary_y=True,
                    
                )  
        
    fig2.add_trace(
    go.Scatter(x=plot_current_cycle3['cycle_days'], 
                y=plot_current_cycle3[y_variable3], 
                name= "Current Cycle " + labels_dict[y_variable3],
                line=dict(color="#FFB983")
                
                
                ),
        secondary_y=False,
        
    ) 
    fig2.add_trace(
    go.Scatter(x=plot_current_cycle4['cycle_days'], 
                y=plot_current_cycle4[y_variable4], 
                name= "Current Cycle " + labels_dict[y_variable4],
                line=dict(color="#C900BB")
                
                
                ),
        secondary_y=True,
        
    )
    fig2.update_yaxes(title_text=labels_dict[y_variable3], secondary_y=False)
    fig2.update_yaxes(title_text=labels_dict[y_variable4], secondary_y=True)

if show_raleos & len(cycle_raleos)>0:
    for i in cycle_raleos['cycle_days']:
        fig.add_vline(x =i, line_width = 2, line_dash = "dash", line_color = 'red', annotation_text= 'Raleo',)
        if second_graph:
            fig2.add_vline(x =i, line_width = 2, line_dash = "dash", line_color = 'red', annotation_text= 'Raleo',)
   
    # Add figure title
fig.update_layout(
        title_text="Pond Diagnostics",
        yaxis2=dict(
            side="right",
            tickmode="sync",
        ),
    )

fig2.update_layout(
        yaxis2=dict(
            side="right",
            tickmode="sync",
        ),
    )

    # Set x-axis title
fig.update_xaxes(title_text="Cycle Days")
fig2.update_xaxes(title_text="Cycle Days")

    # Set y-axes titles
fig.update_yaxes(title_text=labels_dict[y_variable1], secondary_y=False)
fig.update_yaxes(title_text=labels_dict[y_variable2], secondary_y=True)


    

st.plotly_chart(fig, use_container_width=True)

if second_graph:
    st.plotly_chart(fig2, use_container_width=True)
if show_raleos & len(cycle_raleos)>0:
    cycle_raleos.rename(columns = {'Fecha':'Date','cycle_days':'Cycle Days','CantidadCosechada':'Quantity','PesoPromedio':'Average Weight'}, inplace = True)
    
    cycle_raleos[['PKCosecha','Date','Cycle Days','Quantity','Average Weight' ]]


    # Using object notation


