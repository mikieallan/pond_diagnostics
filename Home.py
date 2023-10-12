import streamlit as st
import pandas as pd 
import numpy as np
st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("Active Cycles:")

st.sidebar.success("Select a demo above.")
active_cycles = pd.read_csv('active_cycles.csv')

active_cycles.sort_values('PesoPromedio2', inplace = True)
active_cycles[['pondName', 'PesoPromedio2', 'cycle_days', 'density_ha']]