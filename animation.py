%pip install plotly
conda install plotly.express


import plotly.express as px
import pandas as pd
#%%
df = pd.read_csv(r"df_animation.csv")
#%%
df.tail()
#%% Month, CUSTOMER and GROUP
px.scatter(df, x='COUNTRY', y='GRANDTOTAL')
#%%
px.scatter(data_frame=df, x='gdpPercap', y='lifeExp', size='pop')
#%%
px.scatter(data_frame=df,
           x='gdpPercap',
           y='lifeExp',
           size='pop',
           hover_name='country',
           color='country')
#%%
px.scatter(data_frame=df,
           x='gdpPercap',
           y='lifeExp',
           size='pop',
           hover_name='country',
           color='country',
           animation_frame='year',
           animation_group='country')
#%%
px.scatter(data_frame=df,
           x='gdpPercap',
           y='lifeExp',
           size='pop',
           hover_name='country',
           color='country',
           animation_frame='year',
           animation_group='country',
           range_x=[0,50000],
           range_y=[20,90])