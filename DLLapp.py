import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

import plotly.graph_objs as go
import plotly.express as px

import numpy as np
from numpy import cos, sin, sqrt
import torch 

from DLL_training import Training

# colors = {
#     'background': "#282b38",
#     'text': '#fec036', 
#     'statslider-lab' : "#18df9d",
#     'statslider-mark' : "#6d6f6f",
#     'markers' : '#e5c472',
#     'scat-marker-line' : 'white',
# }

num_layer_max = 5

X = []
minx = None
maxx = None
Xtrain = []
Ytrain = []
Ymodel = []
hdn_values = dict()

trainingSizes = [0, 5, 10, 50, 100, 250, 500, 1000, 2500, 5000, 10000]

colors = {'background' : '#000000', 
          'grid' : '#767676', 
          'text' : '#ECDBCC',
          'disabled' : '#636161'}

def readGivenFunction(inpt) :
    
    f = lambda x : eval(inpt)
    
    msg = ""
    
    try : f(0)
    except : msg = "Non Valid Formula"
    
    return f, msg



def createLayersDisplay(num) : 
    
    out = [
        html.Div([  
            'IN ->'
            ], 
                 style={'width' : str(38 - 6*num) + '%', 
                        'display': 'inline-block',
                        'textAlign': 'right',
                        #'font-size': '150%',
                        'color' : 'white',
                        }
                 )
        ]
    children = []
    for n in range(num) : 
        div = html.Div(
            children = [
                html.Div(
                    children = [
                        html.Label(id='layer-{}'.format(n+1),
                           style={
                               #'white-space': 'pre',
                               #'display': 'inline-block',
                               #'width': '24%',
                               'textAlign': 'center',
                               'font-size' : '110%',
                               'color': 'white'
                               }),
                        ]),
                html.Div(
                    [dcc.Slider(id='hdn-layer-size-{}'.format(n+1), 
                                           min=1,
                                           max=8,
                                           value=3,
                                           #marks={i: {'label' : None} for i in range(8+1)},
                                           step=1,
                                           vertical=True, 
                                           verticalHeight = 80,
                                           )],
                    style = {
                             'padding-left':'44%',
                             },
                    ),
                ], 
            style = {'width' : str(100/num) + '%',
                     "color" : "white",
                     #'textAlign': 'center',
                     'display': 'inline-block',
                     
                     }
            )
        children.append(div)
        
    out.append(html.Div(children, 
                        style = {'width' : str(14*num) + '%',
                     "color" : "white",
                     'vertical-align': 'middle',
                     'display': 'inline-block',
                     
                     }),
                
               )
    out.append(html.Div(['-> OUT'], 
                 style={'color' : 'white',
                        'display': 'inline-block',
                        'textAlign': 'left',
                        'width' : str(38 - 6*num) + '%', 
                        #'vertical-align': 'middle',
                        
                        }
                 ))
    
    return out       
    

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(
    
    style = {'backgroundColor' : colors['background'],
             'marginTop': 0, 
             'marginLeft': 0,
             'marginBottom': 0,
             'font-family' : "Courier New",#'Trebuchet MS',
             },

    children = [ 
        html.Div(
            #className="Banner",
            style = {'padding' : '3px',
                     'borderBottom' : 'solid',
                     'borderColor' : "white",
                     "borderWidth" : "1px",
                     # 'display': 'inline-block',
                     #'textAlign' : 'right',
                     },
            children=[
                html.Div(
                    #className="container scalable",
                    children=[
                        html.Div( 
                            children = [
                                html.H4(
                                    "DEEP LEARNING LABORATORY",
                                    style={'white-space': 'pre',
                                           'vertical-align' : 'middle',
                                           #'display': 'inline-block',
                                           'padding-left' : '1.5%',
                                        #"text-decoration": "none",
                                        "color": "#CCDEEC", 
                                        "width" : "50%"
                                            },
                                    )
                                ],
                                    style={'white-space': 'pre',
                                           'vertical-align' : 'middle',
                                           'display': 'inline-block',
                                           'textAlign' : 'left',
                                           "width" : "49%"
                                            },
                            ),
                        html.Div(
                            children = [
                                html.A(
                                    html.Button('Reset', 
                                                style = {
                                                    'color' : '#CCDEEC',
                                                    'borderColor' : '#CCDEEC'}),
                                    href='/', 
                                    )
                                ],
                            style = {'width' : '50%',
                                     #'padding-right' : '0.8%',
                                     'textAlign': 'right',
                                     'display': 'inline-block',})
                        ,
                    ],
                    style = {
                        'textAlign' : 'center',}
                )
            ],
        ),
    
    html.Div(
        id = 'body',
        style = {'padding' : '3px',
                     'borderBottom' : 'solid',
                     'borderColor' : "white",
                     "borderWidth" : "1px",
                     },
        className="container-scalable",
            children=[
                html.Div(
                    children = [
                        
                        html.Div(
                            children = [
                                html.H6("Write your function",
                                        style = {"color" : colors['text']}),
                                html.Div(
                                    children = [
                                        dcc.Input(id='my-input', value='x**2*cos(2*x+3)', type='text'),
                                        html.Label(id='warning', 
                                                     style = {'width' : '90%',
                                                              "color" : "white",
                                                              'textAlign': 'center',
                                                              'display': 'inline-block',
                                                              }), 
                                        ],
                                        style = {'width' : '49%',
                                                 "color" : "white",
                                                 'textAlign': 'center',
                                                 'display': 'inline-block',
                                                 }
                                       ),
                                html.Div('x will always be your variable. You can use sum and multiplication and power, cos and sin, and of course sqrt.',
                                         style = {'width' : '90%',
                                                  "color" : "grey",
                                                  'textAlign': 'center',
                                                  'display': 'inline-block',
                                                              }),     
                                html.Div('Have fun !',
                                         style = {'width' : '80%',
                                                  "color" : "grey",
                                                  'textAlign': 'center',
                                                  'display': 'inline-block',
                                                              }),     
                                ],
                                style = {"color" : colors['text'],
                                         'textAlign': 'center',
                                         'borderBottom' : 'solid',
                                         'padding' : '3px',
                                         'borderColor' : "white",
                                         "borderWidth" : "0px",
                                         },
                                   ),
                             html.Div(
                                 children = [
                                    html.Div(
                                        children = [
                                            html.Div(
                                                children = [
                                                    html.Label(
                                                            '  Abscissa interval : ',
                                                           style={
                                                               'white-space': 'pre',
                                                               'display': 'inline-block',
                                                               'width': '75%',
                                                               'textAlign': 'left',
                                                               'font-size' : '110%',
                                                               'color': 'white'
                                                                }
                                                               ),
                                                    html.Label(
                                                            id='interval-label',
                                                           style={
                                                               'white-space': 'pre',
                                                               'display': 'inline-block',
                                                               'width': '24%',
                                                               'textAlign': 'right',
                                                               'font-size' : '110%',
                                                               'color': "white"
                                                                }
                                                           ),    
                                                    ],
                                                ),
                                            
                                            dcc.RangeSlider(
                                                id="input-interval",
                                                min=-100,
                                                max=100,
                                                #marks={0: {'label' : str(0), 'style' : {'color' : colors['statslider-mark']}},
                                                #       M: {'label' : str(M), 'style' : {'color' : colors['statslider-mark']}}},
                                                step=1,
                                                updatemode='drag',
                                                #disabled = True,
                                                value=[-5, 10],
                                                pushable=1,
                                                ),
                                            
                                            ],
                                        style = {"width" : '100%',
                                                 'vertical-align': 'middle',
                                                 'display': 'inline-block',
                                                 }
                                        
                                        ),
                                     ],
                                 style = {"color" : "white",
                                                 'textAlign': 'center',
                                                 'borderBottom' : 'solid',
                                                 'padding' : '3px',
                                                 'borderColor' : "white",
                                                 "borderWidth" : "1px",
                                                 'padding-top':'4.5%',
                                                 },),
                            


                            html.Div(
                                children = [
                                    
                                    html.H6("Create your training set",
                                        style = {"color" : colors['text']}),

                                    
                                         
                                    
                                    html.Div(
                                        children = [
                                            html.Div(
                                                children = [
                                                    html.Div(
                                                        children = [
                                                            
                                                            html.Div(
                                                                children = [
                                                                    html.Label('  Size of training data :', 
                                                                       style={
                                                                                'white-space': 'pre',
                                                                                'width': '75%',
                                                                                'display': 'inline-block',
                                                                                'textAlign': 'left',
                                                                                'font-size' : '110%',
                                                                                'color': 'white',
                                                                                            }),
                                                                    
                                                                    html.Label(id='output-trainingSize', 
                                                                       style={
                                                                       'white-space': 'pre',
                                                                       'display': 'inline-block',
                                                                       'width': '24%',
                                                                       'textAlign': 'right',
                                                                       'font-size' : '110%',
                                                                       'color': 'white'
                                                                        }
                                                                       ),
                                                                    ],
                                                                ),
                                                            
                                                            
                                                            html.Div(
                                                                dcc.Slider(
                                                                    id='training-size',
                                                                    min=0,
                                                                    max=len(trainingSizes)-1,
                                                                    value=0,
                                                                    disabled = False,
                                                                    marks={i: {'label' : None, 'style' : {'color' : 'white'}} for i, s in enumerate(trainingSizes)},
                                                                    step=1,
                                                                ),
                                                                style={'width': '100%', 
                                                                       'textAlign': 'center',
                                                                       #'float': 'left', 
                                                                       'display': 'inline-block'
                                                                               }
                                                                ),  
                                                            
                                                            ], 
                                                        
                                                        ),
                                                    html.Div(
                                                        children = [
                                                            html.Label('  Noise :', 
                                                                       style={
                                                                        'white-space': 'pre',
                                                                        'width': '75%',
                                                                        'display': 'inline-block',
                                                                        'textAlign': 'left',
                                                                        'font-size' : '110%',
                                                                        'color': 'white',
                                                                                    }),
                                                            html.Label(id='output-noise', 
                                                                       style={
                                                               'white-space': 'pre',
                                                               'display': 'inline-block',
                                                               'width': '24%',
                                                               'textAlign': 'right',
                                                               'font-size' : '110%',
                                                               'color': 'white'
                                                                }
                                                               ),
                                                            
                                                            ]),
                                                    
                                                    dcc.Slider(
                                                        id='noise',
                                                        disabled = False,
                                                        min=0,
                                                        max=100,
                                                        value=20,
                                                        #marks={'': {'label' : str(per) + "%" if per==0 or per==100 or per==50 else None, 'style' : {'color' : 'white'}} for per in np.arange(0, 101, 5)},
                                                        step=1,
                                                    ),
                                                    
                                                ], 
                                        style={'width': '75%', 
                                               'textAlign': 'center',
                                               #'float': 'left', 
                                               'display': 'inline-block'
                                                       }
                                            ),
                                            ]),
                                    
                                    
                                    ],
                                ),
                                
                                ],
                            style={'width': '29%', 
                                   'vertical-align': 'top',
                                   'borderRight' : 'solid',
                                         'padding' : '0px',
                                         'borderColor' : "white",
                                         "borderWidth" : "1px",
                                   #'float': 'left', 
                                   'textAlign': 'center',
                                   'display': 'inline-block'},
                            ),
                        
                
                        html.Div(
                            id='div-graphs',
                            children=[
                                dcc.Graph(
                                    id='graph',
                                    figure=dict(
                                        layout=dict(
                                            #plot_bgcolor='#5f5de2', 
                                            #paper_bgcolor='#5f5de2'
                                            )
                                        ),
                                    )
                                ],
                            style={'width': '70%', 
                                   #'vertical-align': 'middle',
                                   'textAlign': 'center',
                                   'display': 'inline-block'},
                            
                            ), 
                        ],                    
                    ),
                    html.Div(
                        children = [
                                
                            html.Div(
                                children = [
                                    html.Div(
                                        children = [
                                            
                                            html.H6("Tune your deep neural network",
                                                    style = {"color" : colors['text']}),
                                            
                                            html.Div(
                                                children = [
                                                    html.Div(
                                                        children = [
                                                            # html.Label(
                                                            # 'Number of hidden layers : ',
                                                            #    style={
                                                            #        'white-space': 'pre',
                                                            #        'display': 'inline-block',
                                                            #        'width': '75%',
                                                            #        'textAlign': 'left',
                                                            #        'font-size' : '110%',
                                                            #        'color': 'white'
                                                            #         }
                                                            #        ),
                                                            html.Label(
                                                                    id='num-lyrs-display',
                                                                   style={
                                                                       'white-space': 'pre',
                                                                       'display': 'inline-block',
                                                                       #'width': '24%',
                                                                       'textAlign': 'right',
                                                                       'font-size' : '110%',
                                                                       'color': 'white'
                                                                        }
                                                                   ),    
                                                            ],
                                                        style = {
                                                            'display': 'inline-block',
                                                            'width': '66%',
                                                            'textAlign': 'right',
                                                            'vertical-align': 'bot',
                                                            }
                                                        ),
                                                    
                                                    html.Div( 
                                                        children = [
                                                            dcc.Slider(id='num-lyrs',
                                                       min=1,
                                                       max=num_layer_max,
                                                       marks={i: {'label' : None} for i in range(6)},
                                                       value=2,
                                                       step=1,
                                                        ),
                                                            ], 
                                                         style={'width' : '30%',
                                                               'textAlign' : 'left',
                                                                'padding' : '6px',
                                                               'display': 'inline-block',
                                                               'vertical-align': 'top',
                                                               }
                                                         )                                                    
                                                    ],
                                                style={'width' : '65%',
                                                       #'textAlign' : 'center',
                                                       'display': 'inline-block',
                                                       }
                                                ),
                                            
                                            ],
                                       
                                        ),
                                    
                                    html.Div( 
                                        id='lyrs-display',
                                        ),
                                    
                                    html.Div(
                                        children = [ 
                                            html.Div([
                                                html.Button('Count',
                                                            id='num-params-check', 
                                                            style = {
                                                                "font" : "Courier New",
                                                                "color" : "grey",
                                                                #'vertical-align': 'top',
                                                                }),
                                                ],
                                                style = {
                                                    'width' : '15%',
                                                  'textAlign': 'right',
                                                  'vertical-align': 'top',
                                                  'display': 'inline-block',
                                                              }),
                                            html.Div(id='verify-output', 
                                                     style = {'width' : '15%', 
                                                              'textAlign': 'center',
                                                              #'padding-left' : '20px',
                                                              'display': 'inline-block',
                                                              'color' : 'white', 
                                                              'padding' : '6px',}),
                                            
                                            ],
                                        style = {
                                   #'vertical-align': 'top',
                                   'borderBottom' : 'solid',
                                         'padding' : '3px',
                                         #'textAlign': 'left',
                                         'borderColor' : "white",
                                         "borderWidth" : "1px"}
                                        ), 
                                    
                                     html.Div(
                                            children = [
                                                
                                                html.Div(
                                                    children = [
                                                        
                                                        html.H6("Training parameters",
                                                                style = {"color" : colors['text']}),
                                                        
                                                        html.Div(id='epochs-display',
                                                         style = {
                                                             'textAlign': 'right',
                                                             'width': '40%',
                                                             'color' : 'white',
                                                             'display': 'inline-block',
                                                             'vertical-align': 'top',}),
                                                html.Div(
                                                    children = [
                                                        dcc.Slider(id='num-epo',
                                                        min=5000,
                                                        max=100000,
                                                        #marks={i: {'label' : None} for i in range(6)},
                                                        value=10000,
                                                        step=1000,
                                                        ),
                                                        ],
                                                    style = {'textAlign': 'left',
                                                             'width': '50%',
                                                             'color' : 'white', 
                                                             'display': 'inline-block',
                                                             'padding' : '6px',}
                                                    ),
                                                
                                                    ]),
                                                
                                                
                                                
                                                ], 
                                            style = {'padding' : '6px',
                                                     'textAlign': 'center',
                                                     #'width': '65%',
                                                     #'display': 'inline-block',
                                                     #'vertical-align': 'middle'
                                                     }),
                                    
                                    ],
                                style={'width': '45%',
                                       'padding' : '3px',
                                                 'borderRight' : 'solid',
                                                 'borderColor' : "white",
                                                 "borderWidth" : "1px",
                                   'vertical-align': 'top',
                                   'textAlign': 'center',
                                   'display': 'inline-block'},
                                ), 
                            
                            html.Div([
                                html.Div(
                                    children = [
                                        
                                            html.H6("Train your model",
                                                        style = {"color" : colors['text']}),
                                            
                                            html.Div(
                                                children = [
                                                    
                                                    html.Div(
                                                        children = [
                                                           
                                                            
                                                            html.Div(
                                                                children = [
                                                                    html.Button('Validate Settings',
                                                                        id='Validate',
                                                                        n_clicks=0,
                                                                        #disabled = True,
                                                                        style = {'color' : '#99DBB7',
                                                                                 'borderColor' : '#99DBB7', }
                                                                        ),
                                                                    ],
                                                                style = {#'padding' : '10px',
                                                                    #'padding-top':'10%',
                                                                            'textAlign': 'center',
                                                                            "color" : colors['disabled'],
                                                                            'width': '30%',
                                                                            'display': 'inline-block',
                                                                            'vertical-align': 'middle',
                                                                            }),
                                                            html.Div(id='validate-display', 
                                                                     style={'color' : 'white'})
                                                            
                                                            ], 
                                                        style = {'textAlign': 'center',
                                                                 
                                                                 }
                                                        ),
                                                    #html.Div(id='verify-training'),
                                                    
                                                    ], 
                                                style = {#'height' : '100px',
                                                         'vertical-align': 'middle',})
                                                
                                                ],
                                    style = {'height' : '365px',}
                                    ),
                                
                                html.Div(
                                    children = [
                                        html.Button('Plot model',
                                                    n_clicks = 0,
                                                    id='plot-model', 
                                                    #disabled=True,
                                                    style = {
                                                        'textAlign': 'center',
                                                        "color" : colors['disabled'],
                                                        'borderColor' : colors['disabled'],
                                                        #'vertical-align': 'top',
                                                        }),
                                        ], 
                                    style = {'position' : 'relative',
                                             'bottom': '2px',
                                             'padding' : '3px',
                                             #'height' : '418px',
                                             'vertical-align': 'bot',
                                             })
                                
                                ], 
                                style={'width': '54%',
                                       'padding' : '3px',
                                       #'height' : '418px',
                                                 #'borderLeft' : 'solid',
                                                 #'borderColor' : "white",
                                                 #"borderWidth" : "1px",
                                   
                                   'textAlign': 'center',
                                   'display': 'inline-block'},)
                                
                                
                            ],
                        
                        )


        
                ]
        )

# ====================================================================================================
# ====================================================================================================
# Write your function 
# ====================================================================================================
# ====================================================================================================
@app.callback(
    Output(component_id='warning', component_property='children'),
    [Input(component_id='my-input', component_property='value')]
)
def update_warning(input_formula):
    _, msg = readGivenFunction(input_formula)
    
    if msg == "" : 
        return html.Div('Valid Formula')
    
    else :
        return html.Div(['{}'.format(msg)], 
                         style={'color':'red'})

@app.callback(
    Output(component_id='interval-label', component_property='children'),
    [Input(component_id='input-interval', component_property='value')]
)
def update_interval(interval):
    return '[{} ; {}]        '.format(interval[0], interval[1])

        
# ====================================================================================================
# ====================================================================================================
# Training set
# ====================================================================================================
# ====================================================================================================

@app.callback(
    Output(component_id='output-trainingSize', component_property='children'),
    [Input(component_id='training-size', component_property='value')]
)
def update_training_size(s):
    return '{}  '.format(trainingSizes[s])

@app.callback(
    Output(component_id='output-noise', component_property='children'),
    [Input(component_id='noise', component_property='value')]
)
def update_noise(noise):
    return '{}%  '.format(noise)

# ====================================================================================================
# ====================================================================================================
# Deep neural network tuning
# ====================================================================================================
# ====================================================================================================

@app.callback(
    Output(component_id='num-lyrs-display', component_property='children'),
    [Input(component_id='num-lyrs', component_property='value')]
)
def update_num_lyrs(num):
    return 'Number of hidden layers : {}'.format(num)

@app.callback(
    Output(component_id='lyrs-display', component_property='children'),
    [Input(component_id='num-lyrs', component_property='value')]
)
def display_hdn_layers(num):
    global hdn_values
    hdn_values = dict()
    return createLayersDisplay(num)

# ====================================================================================================
# hidden layers tuning

@app.callback(
Output(component_id='layer-1', component_property='children'),
[Input(component_id='hdn-layer-size-1', component_property='value')]
)
def display_hdn_layers_size(ls):
    global hdn_values
    hdn_values[1] = 2**ls
    return '{}'.format(2**ls)

@app.callback(
Output(component_id='layer-2', component_property='children'),
[Input(component_id='hdn-layer-size-2', component_property='value')]
)
def display_hdn_layers_size(ls):
    global hdn_values
    hdn_values[2] = 2**ls
    return '{}'.format(2**ls)

@app.callback(
Output(component_id='layer-3', component_property='children'),
[Input(component_id='hdn-layer-size-3', component_property='value')]
)
def display_hdn_layers_size(ls):
    global hdn_values
    hdn_values[3] = 2**ls
    return '{}'.format(2**ls)

@app.callback(
Output(component_id='layer-4', component_property='children'),
[Input(component_id='hdn-layer-size-4', component_property='value')]
)
def display_hdn_layers_size(ls):
    global hdn_values
    hdn_values[4] = 2**ls
    return '{}'.format(2**ls)

@app.callback(
Output(component_id='layer-5', component_property='children'),
[Input(component_id='hdn-layer-size-5', component_property='value')]
)
def display_hdn_layers_size(ls):
    global hdn_values
    hdn_values[5] = 2**ls
    return '{}'.format(2**ls)
  

@app.callback(
Output(component_id='verify-output', component_property='children'),
[Input(component_id='num-params-check', component_property='n_clicks')]
)
def display_verify(nc):
    hdn_values_lst = list(hdn_values.values())
    num = 1 * hdn_values_lst[0]
    for i in range(len(hdn_values_lst)-1) : num += hdn_values_lst[i] * hdn_values_lst[i+1]
    num +=  hdn_values_lst[-1] * 1
    
    if num <= 66048 : 
        return html.Div([html.Div('{}/66048'.format(num)),
                        ], 
                        style = {'color' : 'green'})
    else : 
        return html.Div(['{}/66048'.format(num)], 
                        style = {'color' : 'red'})

# ====================================================================================================
# ====================================================================================================
# Training
# ====================================================================================================
# ====================================================================================================

@app.callback(
    Output(component_id='epochs-display', component_property='children'),
    [Input(component_id='num-epo', component_property='value')]
)
def update_num_epochs(num):
    return 'Number of epochs : {}'.format(num)

@app.callback(
    Output(component_id='validate-display', component_property='children'),
    [Input(component_id='Validate', component_property='n_clicks')],
    [State('training-size', 'value')]
)
def display_validate(nc, ts):
    
    msg1 = ""
    msg2 = ""
    
    if ts == 0 : 
        msg1 += "You have no training data."
        
    hdn_values_lst = list(hdn_values.values())
    num = 1 * hdn_values_lst[0]
    for i in range(len(hdn_values_lst)-1) : num += hdn_values_lst[i] * hdn_values_lst[i+1]
    num +=  hdn_values_lst[-1] * 1
    
    if num > 66048 : 
        msg2 += "Your network is too big."
        
    if not msg1 + msg2 == "" : 
    
        return html.Div(
                children = [
                    html.Div([msg1], style={'color' : 'red'}),
                    html.Div([msg2], style={'color' : 'red'})])
    else : 
        
        return html.Div([
                    html.Div([
                        html.Button('Run !',
                           id='training', 
                           style = {
                               #'width': '30%', 
                               #'textAlign': 'center',
                               'color' : '#99DBB7',
                               'borderColor' : '#99DBB7',
                               
                               }),
                        html.Div(
                             children = [
                                 dcc.Loading(
                                    id="loading-training",
                                    type="dot",
                                    color = '#99BEDB',
                                    #children=[html.Div(id="loading-output-1")],
                                    ),
                                 ],
                             style={'padding' : '50px'}
                             ),
                        ],
                        style={'display': 'inline-block',
                               'width': '15%',
                               'padding-top' : '2%',
                               'vertical-align': 'top',
                               'textAlign': 'center',}),
                    
                    html.Div(
                            children=[
                                dcc.Graph(
                                    id='graph-loss',
                                    config = {
                                           'scrollZoom': True,
                                           'doubleClick' :'reset',
                                           'displayModeBar' : False,
                                           'displaylogo':  False,
                                            'modeBarButtonsToRemove' : ['zoom2d', 
                                                                        #'sendDataToCloud', 
                                                                        'lasso2d', 
                                                                        'autoScale2d',
                                                                        'hoverClosestCartesian', 
                                                                        'hoverCompareCartesian',
                                                                        'toggleSpikelines',
                                                                        'select2d',
                                                                        'resetScale2d',
                                                                        'zoomIn2d',
                                                                        'zoomOut2d'
                                                                        ]
                                   },
                                    ),
                                dcc.Interval(
                                    id='interval-component',
                                    interval=1*500, # in milliseconds
                                    n_intervals=0
                                ),
                                ],
                            style={'width': '85%', 
                                   #'padding-left':'20%',
                                   #'height' : 0,
                                   'vertical-align': 'middle',
                                   'textAlign': 'center',
                                   'display': 'inline-block'
                                   },
                            
                            ),
                    ], 
            style = {'vertical-align': 'middle',
                     #'height' : '800px',
                     'textAlign': 'center',
                     'padding-top' : '6%'
                     })


losses = []
epochs = []

@app.callback(
Output(component_id='training', component_property='style'),
[Input(component_id='training', component_property='n_clicks')],
)
def change_run_button_color(nc) : 
    if nc > 0 : 
        return {'color' : colors['disabled'],
                'borderColor' : colors['disabled']}
    else : 
        return {'color' : '#99DBB7',
                'borderColor' : '#99DBB7'}

@app.callback(
Output(component_id='loading-training', component_property='children'),
[Input(component_id='training', component_property='n_clicks')],
[State(component_id='num-epo', component_property='value')]
)
def do_training(nc, ep):
    
    if int(nc) > 0 : 
        
        model_params = [1] +  list(hdn_values.values()) + [1]
        
        T = Training(Xtrain, Ytrain, model_params)
        #T.epochs(ep)
        
        for epoch in range(ep) :
            T.train()
            if (epoch + 1) % 1000 == 0 :
                global losses
                losses = T.compute_global_loss(epoch)
                global epochs
                epochs.append(epoch+1)
        
        global Ymodel 
        Ymodel = T.model(torch.Tensor(X)).squeeze().tolist()
        
        return html.Div([''], 
                             style = {'color' : '#99BEDB'})

@app.callback(
Output('plot-model', 'style'),
[Input('loading-training', 'children')]
)
def update_plotmodel_button_color(chil) : 
    if not chil == None : 
        return {'color' : '#99DBB7',
                'borderColor' : '#99DBB7'}
    
# @app.callback(
# Output('plot-model', 'disabled'),
# [Input('loading-training', 'children')]
# )
# def update_plotmodel_button_disabled(chil) : 
#     if not chil == None : 
#         return False
    
# @app.callback(
# Output('plot-model', 'style'),
# [Input('plot-model', 'n_clicks')]
# )
# def update_plotmodel_button_color2(nc) : 
#     if nc > 0 : 
#         return {'color' : 'grey',
#                 'borderColor' : 'grey'}
#     else : 
#         return
        
        
@app.callback(
Output('graph-loss', 'figure'),
[Input('interval-component', 'n_intervals')]
)
def display_training(nc):
    
    if nc == 0 : 
        global epochs
        epochs = []
        global losses
        losses = []
    
    data = go.Scatter(         
                x=epochs,
                y=losses,
                mode='markers+lines',
                hoverinfo='skip',
                marker=dict(color="#99BEDB"),
                )
        
    layout = go.Layout(
                xaxis = {'title':'Epochs', 
                         'color' : 'white', 
                         'gridcolor' : colors['grid']},
                yaxis = {'title':'MSE', 
                         'color' : 'white',
                         'gridcolor' : colors['grid'], 
                         'range' : [0, 1.1 * max([1] + losses)]},
                plot_bgcolor=colors['background'], 
                paper_bgcolor=colors['background'],
                showlegend=False,
                font=dict(
                    family="Courier New",
                    #size=18,
                ),
                #dragmode='pan',
                hovermode='closest',
                autosize=True,
                height = 200,
                transition = {'duration': 200,
                              'easing': 'cubic-in-out',
                              },
                margin={'t': 10,'l':50,'b':0,'r':50},
                )
     
    figure = go.Figure(data=data, layout=layout)
    
    return figure
    
@app.callback(
Output(component_id='my-input', component_property='disabled'),
[Input(component_id='training', component_property='n_clicks')]
)
def disable_formula(nc):  
    if int(nc) > 0 : 
        return True

@app.callback(
Output(component_id='input-interval', component_property='disabled'),
[Input(component_id='training', component_property='n_clicks')]
)
def disable_interval(nc):  
    if int(nc) > 0 : 
        return True
    
@app.callback(
Output(component_id='training-size', component_property='disabled'),
[Input(component_id='training', component_property='n_clicks')]
)
def disable_ts(nc):  
    if int(nc) > 0 : 
        return True
    
@app.callback(
Output(component_id='noise', component_property='disabled'),
[Input(component_id='training', component_property='n_clicks')]
)
def disable_noise(nc):  
    if int(nc) > 0 : 
        return True
    
@app.callback(
Output(component_id='num-lyrs', component_property='disabled'),
[Input(component_id='training', component_property='n_clicks')]
)
def disable_nlyrs(nc):  
    if int(nc) > 0 : 
        return True
    
@app.callback(
Output(component_id='hdn-layer-size-1', component_property='disabled'),
[Input(component_id='training', component_property='n_clicks')]
)
def disable_1lyrs(nc):  
    if int(nc) > 0 : 
        return True
    
@app.callback(
Output(component_id='hdn-layer-size-2', component_property='disabled'),
[Input(component_id='training', component_property='n_clicks')]
)
def disable_2lyrs(nc):  
    if int(nc) > 0 : 
        return True
    
@app.callback(
Output(component_id='hdn-layer-size-3', component_property='disabled'),
[Input(component_id='training', component_property='n_clicks')]
)
def disable_3lyrs(nc):  
    if int(nc) > 0 : 
        return True
    
@app.callback(
Output(component_id='hdn-layer-size-4', component_property='disabled'),
[Input(component_id='training', component_property='n_clicks')]
)
def disable_4lyrs(nc):  
    if int(nc) > 0 : 
        return True
    
@app.callback(
Output(component_id='hdn-layer-size-5', component_property='disabled'),
[Input(component_id='training', component_property='n_clicks')]
)
def disable_5lyrs(nc):  
    if int(nc) > 0 : 
        return True
    
@app.callback(
Output(component_id='num-epo', component_property='disabled'),
[Input(component_id='training', component_property='n_clicks')]
)
def disable_nu_epochs(nc):  
    if int(nc) > 0 : 
        return True
    
@app.callback(
Output(component_id='Validate', component_property='style'),
[Input(component_id='Validate', component_property='n_clicks')],
[State('training-size', 'value')]
)
def disable_validate_trainingCol(nc, ts):  
    msg1 = ""
    msg2 = ""
    
    if ts == 0 : 
        msg1 += "You have no training data."
        
    hdn_values_lst = list(hdn_values.values())
    num = 1 * hdn_values_lst[0]
    for i in range(len(hdn_values_lst)-1) : num += hdn_values_lst[i] * hdn_values_lst[i+1]
    num +=  hdn_values_lst[-1] * 1
    
    if num > 66048 : 
        msg2 += "Your network is too big."
        
    if not msg1 + msg2 == "" : 
        return {'color' : '#99DBB7',
                'borderColor' : '#99DBB7', }
    
    else : 
        return {'color': colors['disabled'], 
                'borderColor':colors['disabled']}
        
        
    # if nc > 0 and : 
    #     return {'color': colors['disabled'], 
    #             'borderColor':colors['disabled']}
    # else : 
    #     return {'color' : '#99DBB7',
    #             'borderColor' : '#99DBB7', }
    
# @app.callback(
# Output(component_id='Validate', component_property='disabled'),
# [Input(component_id='Validate', component_property='n_clicks')]
# )
# def disable_validate_training(nc):  
#     if nc > 0 : 
#         return True
#     else : 
#         return False


    
    
    
# ====================================================================================================
# ====================================================================================================
# Display Graph
# ====================================================================================================
# ====================================================================================================

@app.callback(
    Output("div-graphs", "children"),
    [Input('my-input', 'value'),
     Input('input-interval', 'value'),
     Input('plot-model', 'n_clicks'), 
     Input('noise', 'value'), 
     Input('training-size', 'value'),
     ], 
    )
def update_graph(input_formula, interval, nc_pm, noise, s):
    
    global minx
    minx = interval[0]
    global maxx
    maxx = interval[1]
    
    fun, msg = readGivenFunction(input_formula)
    
    if nc_pm == 0 :
        global Xtrain
        Xtrain = np.random.uniform(minx, maxx, trainingSizes[s])
    
    
    if msg == "" : 
        
        global X
        X = np.linspace(minx, maxx, 1000)
        Y = [fun(x) for x in X]
        amplitude = max(Y) - min(Y)
        yN = np.random.normal(0, (noise /100)**1.75 * amplitude, trainingSizes[s])
        if nc_pm == 0 : 
            global Ytrain
            Ytrain = [fun(x) + yn for x, yn in zip(Xtrain, yN)]
        
    else : 
        
        #global X
        X = np.linspace(minx, maxx, 1000)
        Y = np.zeros(1000)
        #global Ytrain
        Ytrain = np.zeros(trainingSizes[s])
        
    if nc_pm == 0 : 
        
        
        global Ymodel 
        Ymodel =[]
        
        data = [
            go.Scatter(
                name="Observations",
                x=Xtrain,
                y=Ytrain,
                mode='markers',
                hoverinfo='skip',
                opacity=0.75,
                marker=dict(size=6, 
                            color="#ECCCCE",
                            # line=dict(
                            #     color='white',
                            #     width=0.35)
                            ),
                
                ),
            go.Line(
                name="Reality",
                    x=X,
                    y=Y,
                    line = dict(color="#B73239"),
                    #mode='lines',
                    #width=20,
                    hoverinfo='skip',
                    #color="blue",
                    ),
            ]
        
        
    else : 
        
        data = [
            go.Scatter(
                name="Observations",
                x=Xtrain,
                y=Ytrain,
                mode='markers',
                hoverinfo='skip',
                opacity=0.75,
                marker=dict(size=6, 
                            color="#ECCCCE",
                            # line=dict(
                            #     color='white',
                            #     width=0.35)
                            ),
                
                ),
            go.Line(
                    name="Reality",
                    x=X,
                    y=Y,
                    line = dict(color="#B73239"),
                    #mode='lines',
                    #width=20,
                    hoverinfo='skip',
                    #color="blue",
                    ),
            go.Line(
                name="Model",
                x=X,
                y=Ymodel,
                hoverinfo='skip',
                #opacity=0.75,
                line = dict(width=4, color="#CEECCC", 
                            #dash='dash'
                            ),
                ),
            ]
        
    layout = go.Layout(
            xaxis = {'zeroline':False,
                'color' : 'white', 
                'gridcolor' : colors['grid']},
            yaxis={'zeroline':False,
                   'gridcolor' : colors['grid'],
                   'showticklabels':False},
            font=dict(
                family="Courier New",
                #size=18,
            ),
            #showlegend=False,
            #dragmode='pan',
            hovermode='closest',
            legend=dict(
                traceorder="reversed",
                bordercolor="white",
                borderwidth=1,
                x=0,
                y=1,
                font = dict(
                    size=16,
                    color="white",
                    )
                
            ),
            autosize=True,
            transition = {'duration': 10,
                          #'easing': 'cubic-in-out',
                          },
            margin={'t': 10,'l':50,'b':0,'r':50},
            plot_bgcolor=colors['background'], 
            paper_bgcolor=colors['background']
        )

    figure = go.Figure(data=data, layout=layout)

    
    return html.Div(
            id="graph-container",
            children=[dcc.Loading(
                 className="graph-wrapper",
                 children=dcc.Graph(id="graph", 
                                    figure=figure,
                                    config = {
                                       'scrollZoom': True,
                                       'doubleClick' :'reset',
                                       'displayModeBar' : False,
                                       'displaylogo':  False,
                                        'modeBarButtonsToRemove' : ['zoom2d', 
                                                                    #'sendDataToCloud', 
                                                                    'lasso2d', 
                                                                    'autoScale2d',
                                                                    'hoverClosestCartesian', 
                                                                    'hoverCompareCartesian',
                                                                    'toggleSpikelines',
                                                                    'select2d',
                                                                    'resetScale2d',
                                                                    'zoomIn2d',
                                                                    'zoomOut2d'
                                                                    ]
                               },
                                    ),
             ),
                ],
            #style = {'vertical-align': 'middle'}
        )


if __name__ == '__main__':
    app.run_server(debug=False,dev_tools_ui=False,dev_tools_props_check=False)