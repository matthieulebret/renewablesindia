import streamlit as st
import plotly.express as px
import plotly.graph_objs as go
#import plotly.offline as py

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import datetime
import xlrd

import re
import itertools
import networkx as nx


st.set_page_config('Indian renewables market',layout='wide',initial_sidebar_state="collapsed")


im1,im2,im3 = st.beta_columns(3)
with im1:
    st.image('https://images.unsplash.com/photo-1524492412937-b28074a5d7da?ixid=MXwxMjA3fDB8MHxzZWFyY2h8MXx8aW5kaWF8ZW58MHx8MHw%3D&ixlib=rb-1.2.1&auto=format&fit=crop&w=400&q=60')
with im2:
    st.image('https://images.unsplash.com/photo-1519802772250-a52a9af0eacb?ixid=MXwxMjA3fDB8MHxzZWFyY2h8NHx8aW5kaWF8ZW58MHx8MHw%3D&ixlib=rb-1.2.1&auto=format&fit=crop&w=400&q=60')
with im3:
    st.image('https://images.unsplash.com/photo-1463592177119-bab2a00f3ccb?ixid=MXwxMjA3fDB8MHxzZWFyY2h8MTV8fGluZGlhfGVufDB8fDB8&ixlib=rb-1.2.1&auto=format&fit=crop&w=400&q=60')

st.title('Analysis of Indian renewables market')



st.sidebar.title('Select filters')

#Data acquisition

pfmarket = pd.read_excel('Lenders_135_All_market.xlsx',header=1).iloc[:,:6]


sectorlist = pfmarket['Sector'][0].split('|')


newlist = []
for sector in sectorlist:
    sector = sector.split(':')[0].replace(' ','',1)
    newlist.append(sector)
sectorlist = newlist

sectorlist.insert(0,'Environment')

def extractsector(string):
    allocation = []
    i=0
    for sector in sectorlist:
        if string.find(sector) == -1:
            allocation.append(0)
        else:
            allocation.append(re.findall('[0-9]+',string)[i])
            i=i+1
    return allocation

pfmarket['Allocation']=pfmarket['Sector'].apply(extractsector)

i=0

for sector in sectorlist:
    pfmarket[sectorlist[i]] = pfmarket['Allocation'].apply(lambda x: x[i])
    pfmarket[sectorlist[i]] = pd.to_numeric(pfmarket[sectorlist[i]])/100
    i = i+1


minsize = st.number_input('Minimum invested size filter',min_value=0,value=100)
pfmarkettable = pfmarket[pfmarket['Lent amount  (USD)m']>minsize]
pfmarkettable = pfmarket.drop(['Asset location','Sector','Allocation','Origin','Deal type'],axis=1)

st.subheader('Scroll to the right to see the portfolio splits')
st.write(pfmarkettable.style
    .background_gradient(cmap='viridis',subset=sectorlist)
    .format('{:.2%}',subset=sectorlist))

# pfmarkettable.to_excel('pfmarketindia.xlsx')

for sector in sectorlist:
    pfmarket[sector+' amount'] = pfmarket['Lent amount  (USD)m'] * pfmarket[sector]

st.subheader('Split of Indian pf market - lent amount since 2014')

selectsplit = st.radio('Select split by bank or by sector',['Bank','Sector'],index=1)


treemapdf = pfmarket[['Name','Environment amount','Energy amount','Other amount','Power amount','Renewables amount','Transport amount']]
treemapdf.set_index('Name',inplace=True)
treemapdf = treemapdf.stack()
treemapdf = pd.DataFrame(treemapdf)
treemapdf.reset_index(inplace=True)
treemapdf.columns=['Name','Sector','Lent amount  (USD)m']

if selectsplit == 'Bank':
    fig = px.treemap(treemapdf,path=['Name','Sector'],values='Lent amount  (USD)m')
else:
    fig = px.treemap(treemapdf,path=['Sector','Name'],values='Lent amount  (USD)m')

st.plotly_chart(fig)


#### Data acquisition

deallist = pd.read_excel(r'348__Projects.xlsx',header=1)
bankdeal = pd.read_excel(r'Facilities_69.xlsx',header=1)

deallist = deallist[['Transaction Name','States/provinces']]

#st.write(deallist)

bankdeal.iloc[:,0:2] = bankdeal.iloc[:,0:2].fillna(method='pad')
bankdeal.dropna(axis=0,subset=['Name'],inplace=True)
bankdeal = bankdeal.iloc[:,:11]

bankdeal.rename(columns={'Unnamed: 0':'Bank'},inplace=True)

bankdeal = pd.merge(bankdeal,deallist,left_on='Name',right_on='Transaction Name')

## Selection filter widgets in the sidebar

#selectyear = st.sidebar.multiselect('Select years',(2016,2017,2018,2019,2020),default=[2016,2017,2018,2019,2020])
selectyear = st.sidebar.slider('Select period',min_value=2016,max_value=2021,value=(2016,2021),step=1)
selectstate = st.sidebar.multiselect('Select states',list(bankdeal['States/provinces'].unique()),default=list(bankdeal['States/provinces'].unique()))

bankdeal = bankdeal[bankdeal['Total lent amount (USD)m']>0]
bankdeal['Year'] = pd.DatetimeIndex(bankdeal['Financial close date']).year
bankdeal = bankdeal[bankdeal['Year'].isin(range(selectyear[0],selectyear[1]+1))]
bankdeal = bankdeal[bankdeal['States/provinces'].isin(selectstate)]

bankdeal = bankdeal[['Bank','Origin','Name','Sub-sector','Deal type','Lent amount (USD)m','Total lent amount (USD)m','States/provinces','Year']]

### Data display

st.header('Deals by bank')
mydeal = st.text_input(r'Looking for a particular deal? Search in the textbox below.')
if mydeal == '':
    st.write(bankdeal)
else:
    st.write(bankdeal[bankdeal['Name'].str.contains(mydeal)])

st.header('Deal volume per year')
fig = px.bar(bankdeal,x='Year',y='Lent amount (USD)m',color='Sub-sector',hover_name='Name',template='plotly_white')
fig.update_layout(barmode='stack',xaxis=dict(tickmode='array',tickvals=[2016,2017,2018,2019,2020,2021]))
st.plotly_chart(fig,use_container_width=True,xaxis=dict(showlabel=False))


st.header('Compare deals across two banks')

selectsector = st.radio('Select sub-sector',bankdeal['Sub-sector'].unique(),index=1)

bank1 = st.selectbox('Bank 1',bankdeal['Bank'].sort_values().unique(),index=0)
bank2 = st.selectbox('Bank 2',bankdeal['Bank'].sort_values().unique(),index=1)

banklist = [bank1,bank2]

bankdealfilter = bankdeal[bankdeal['Bank'].isin(banklist)]

trace1 = bankdealfilter[(bankdealfilter['Bank']==bank1) & (bankdealfilter['Sub-sector']==selectsector)]
trace2 = bankdealfilter[(bankdealfilter['Bank']==bank2) & (bankdealfilter['Sub-sector']==selectsector)]

fig = go.Figure(data=[
    go.Bar(name=bank1,x=trace1['Year'],y=trace1['Lent amount (USD)m'],text=trace1['Name']),
    go.Bar(name=bank2,x=trace2['Year'],y=trace2['Lent amount (USD)m'],text=trace2['Name'])
        ])

fig.update_layout(barmode='group',template='plotly_white',xaxis=dict(tickmode='array',tickvals=[2016,2017,2018,2019,2020,2021]))
st.plotly_chart(fig,use_container_width=True)



st.header('Deal volume by bank')
fig = px.bar(bankdeal,x='Bank',y='Lent amount (USD)m',color='Sub-sector',hover_name='Name',template='plotly_white')
fig.update_layout(barmode='stack',xaxis={'categoryorder':'total descending'})
st.plotly_chart(fig,use_container_width=True,xaxis=dict(showlabel=False))
#py.plot(fig,filename='bar.html')

st.header('Deal by deal analysis')
dimensionoptions = ['Country > Bank > Sub Asset Class > Deal detail','Sub Asset Class > Deal detail > Bank','Sub Asset Class > Bank > Deal detail','Sub Asset Class > Deal type > Bank > Deal detail','State > Sub Asset Class > Bank > Deal detail']
selectdimensions = st.selectbox('Select dimensions, the sunburst chart below will adjust.',dimensionoptions,index=0)

if selectdimensions == dimensionoptions[0]:
    st.subheader('Deal split per bank')
    st.text(selectdimensions)
    fig = px.sunburst(bankdeal,path=['Origin','Bank','Sub-sector','Name'],values='Lent amount (USD)m')
elif selectdimensions == dimensionoptions[1]:
    st.header('Split of banks by deal type - club by deal')
    st.text(selectdimensions)
    fig = px.sunburst(bankdeal,path=['Sub-sector','Name','Bank'],values='Lent amount (USD)m')
elif selectdimensions == dimensionoptions[2]:
    st.header('Split of banks by deal type - deals by bank')
    st.text(selectdimensions)
    fig = px.sunburst(bankdeal,path=['Sub-sector','Bank','Name',],values='Lent amount (USD)m')
elif selectdimensions == dimensionoptions[3]:
    st.header('Split of deals by type')
    st.text(selectdimensions)
    fig = px.sunburst(bankdeal,path=['Sub-sector','Deal type','Bank','Name',],values='Lent amount (USD)m')
elif selectdimensions == dimensionoptions[4]:
    st.header('Split of deals by state')
    st.text(selectdimensions)
    fig = px.sunburst(bankdeal,path=['States/provinces','Sub-sector','Bank','Name',],values='Lent amount (USD)m')

st.plotly_chart(fig,use_container_width=True,xaxis=dict(showlabel=False))

st.header('Distribution of deals (NordLB vs Market)')
distribchoice = st.radio('Select distribution axis.',['Total debt size by deal','Ticket size by deal'],index=0)


if distribchoice == 'Total debt size by deal':
    fig = px.histogram(bankdeal,x='Total lent amount (USD)m',y='Total lent amount (USD)m',color='Sub-sector',nbins=20,template='plotly_white')
    if 'Norddeutsche Landesbank Girozentrale (NORD/LB)' in bankdeal['Bank']:
        fig.add_shape(type='line',x0=bankdeal[bankdeal['Bank']=='Norddeutsche Landesbank Girozentrale (NORD/LB)']['Total lent amount (USD)m'].mean(),
                                y0=0,
                                x1=bankdeal[bankdeal['Bank']=='Norddeutsche Landesbank Girozentrale (NORD/LB)']['Total lent amount (USD)m'].mean(),
                                y1=50,
        line=dict(
        color='MediumPurple',
        width=6,
        dash='dot'))
else:
    fig = px.histogram(bankdeal,x='Lent amount (USD)m',y='Lent amount (USD)m',color='Sub-sector',nbins=20,template='plotly_white')
    if 'Norddeutsche Landesbank Girozentrale (NORD/LB)' in bankdeal['Bank']:
        fig.add_shape(type='line',x0=bankdeal[bankdeal['Bank']=='Norddeutsche Landesbank Girozentrale (NORD/LB)']['Lent amount (USD)m'].mean(),
                            y0=0,
                            x1=bankdeal[bankdeal['Bank']=='Norddeutsche Landesbank Girozentrale (NORD/LB)']['Lent amount (USD)m'].mean(),
                            y1=60,
        line=dict(
        color='MediumPurple',
        width=6,
        dash='dot'))

fig.update_layout(barmode = 'overlay')
fig.update_traces(opacity=0.75)
st.plotly_chart(fig,use_container_width=True)


###########################################
############### Network ###################
###########################################


st.header('Network analysis')

period = st.slider("Select years",min_value=2016,max_value=2020,value=(2019,2020),step=1)

bankdeal = bankdeal[(bankdeal['Year']>=period[0])&(bankdeal['Year']<=period[1])]

bankdeal

banklist = bankdeal['Bank'].unique().tolist()
deallist = bankdeal['Name'].unique().tolist()

## This is a list of tuples
bankpairlist = list(itertools.combinations(banklist,2))


def getpairnexus(pair):
    nbcommondeals = 0
    for deal in deallist:
        dealbanklist = bankdeal['Bank'][bankdeal['Name']==deal].tolist()
        if (pair[0] in dealbanklist) and (pair[1] in dealbanklist):
            nbcommondeals += 1
    return nbcommondeals

nbdeals=[]
for pair in bankpairlist:
    nbdeals.append(getpairnexus(pair))

df = pd.DataFrame([[bank[0] for bank in bankpairlist],[bank[1] for bank in bankpairlist],nbdeals]).transpose()
df.columns = ['Bank1','Bank2','Nb deals in common']

df = df[df['Nb deals in common']!=0]

#
# st.write(df.to_csv('nexus.csv'))

# df = pd.read_csv('nexus.csv')
# df = df.iloc[:,1:]
# st.write(df)


#### Get number of deals per bank
dealsperbank = bankdeal.loc[:,['Bank','Name']]
newdf = dealsperbank.groupby('Bank').count()

newdf = pd.DataFrame(newdf)

def getnbdeal(bank):
    nbdeal = newdf.loc[bank,'Name']
    return nbdeal

#### Network graph

renmarket = nx.Graph()

## Add node to each bank



for bank in banklist:
    size = getnbdeal(bank)
    renmarket.add_node(bank,size=getnbdeal(bank))
    # renmarket.add_node(bank,size=np.random.randint(1,5,1))


## For each co-apprearance between two banks, add an edge

############ WORK ON THIS ###########################

for bankpair in bankpairlist:
    try:
        weight = df[(df['Bank1']==bankpair[0])&(df['Bank2']==bankpair[1])]['Nb deals in common'].item()
    except:
        weight = 1
    renmarket.add_edge(bankpair[0],bankpair[1],weight=weight)


## Get positions for the nodes

pos_ = nx.drawing.layout.spring_layout(renmarket)

## Create an edge between node x and node y with a given text and width

def make_edge(x,y,text,width):
    return go.Scatter(x=x,y=y,line = dict(width=width,color='cornflowerblue'),hoverinfo='text',text=([text]),mode='lines')

### For each edge make an edge_trace, append to list

edge_trace = []
for edge in renmarket.edges():
    try:
        width = renmarket.edges()[edge]['weight']
    except:
        width = 0
    char_1 = edge[0]
    char_2 = edge[1]
    x0,y0 = pos_[char_1]
    x1,y1 = pos_[char_2]
    text = char_1 + '--' + char_2 + ': ' + str(renmarket.edges()[edge]['weight'])
    trace = make_edge([x0,x1,None],[y0,y1,None],text,width=width)
    edge_trace.append(trace)

# Make a node trace
node_trace = go.Scatter(x         = [],
                        y         = [],
                        text      = [],
                        textposition = "top center",
                        textfont_size = 10,
                        mode      = 'markers+text',
                        hoverinfo = 'none',
                        marker    = dict(color = [],
                                         size  = [renmarket.nodes()[node]['size'] for node in renmarket.nodes],
                                         line = None))

# For each node in renmarket, get the position and size and add to the node_trace

for node in renmarket.nodes:
    x, y = pos_[node]
    node_trace['x'] += tuple([x])
    node_trace['y'] += tuple([y])
    node_trace['marker']['color'] += tuple(['red'])
    # node_trace['marker']['size'] += renmarket.nodes()[node]['size']
    node_trace['text'] += tuple(['<b>' + node + '</b>'])


# Customize layout
layout = go.Layout(
    paper_bgcolor='rgba(0,0,0,0)', # transparent background
    plot_bgcolor='rgba(0,0,0,0)', # transparent 2nd background
    xaxis =  {'showgrid': False, 'zeroline': False}, # no gridlines
    yaxis = {'showgrid': False, 'zeroline': False}, # no gridlines
)
# Create figure
fig = go.Figure(layout = layout)
# Add all edge traces
for trace in edge_trace:
    fig.add_trace(trace)
# Add node trace
fig.add_trace(node_trace)
# Remove legend
fig.update_layout(showlegend = False)
# Remove tick labels
fig.update_xaxes(showticklabels = False)
fig.update_yaxes(showticklabels = False)
# Show figure

st.plotly_chart(fig)
