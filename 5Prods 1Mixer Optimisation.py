import pandas as pd
import numpy as np
from pyomo.environ import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as mticker
from matplotlib.patches import ConnectionPatch

# Extracting the RM Comppsition Table
data_path = 'https://github.com/robithh/Research_Project/raw/main/RM%20Composition.xlsx'
df = pd.read_excel(data_path, index_col='INGREDIENT')

# Define Variables Name based on DataFrame index and columns
SUB = df.index.tolist()
RM = df.columns.tolist()

# Extract parameters (convert DataFrame to dictionary of dictionaries)
comp_dict = df.to_dict(orient='index')

# Define the model
model = ConcreteModel()

# Define Set - the Mixer RM Composition
model.SUB = Set(initialize=SUB)
model.RM = Set(initialize=RM)

# Define Parameters
def param_init(model, sub, rm):
    return comp_dict[sub][rm]

model.composition = Param(model.SUB, model.RM, initialize=param_init)

# Create Decision Variables (only for RM quantities)
model.Mixer_RM_qty = Var(model.RM, within=NonNegativeReals, bounds=(0, 1))
model.Mixer_Level_A = Var(within=NonNegativeReals, bounds=(0, 1), initialize=1)
model.Mixer_Level_B = Var(within=NonNegativeReals, bounds=(0, 1), initialize=0)
model.Mixer_Level_C = Var(within=NonNegativeReals, bounds=(0, 1), initialize=1)
model.Mixer_Level_D = Var(within=NonNegativeReals, bounds=(0, 1), initialize=0)
model.Mixer_Level_E = Var(within=NonNegativeReals, bounds=(0, 1), initialize=1)
model.prod_A = Var(model.RM, within=NonNegativeReals, bounds=(0, 1))
model.prod_B = Var(model.RM, within=NonNegativeReals, bounds=(0, 1))
model.prod_C = Var(model.RM, within=NonNegativeReals, bounds=(0, 1))
model.prod_D = Var(model.RM, within=NonNegativeReals, bounds=(0, 1))
model.prod_E = Var(model.RM, within=NonNegativeReals, bounds=(0, 1))

# Define Mixer Costs Dictionary
mixer_costs_dict = {
    'RM_01': 800, 'RM_02': 400, 'RM_03': 600, 'RM_05': 1200, 'RM_06': 700,
    'RM_08': 100, 'RM_09': 700, 'RM_10': 1200, 'RM_12': 5000, 'RM_14': 1000,
    'RM_15': 500, 'RM_16': 200, 'RM_18': 800, 'RM_19': 2000, 'RM_20': 6000,
    'RM_21': 900, 'RM_22': 500, 'RM_23': 1100
}

# Initialize Parameters for Mixer Costs
model.mixercosts = Param(model.RM, initialize=mixer_costs_dict, default=0)

# Define Post Mixer Costs Dictionary
post_mixer_costs_dict = {
    'RM_03': 600, 'RM_05': 1200, 'RM_07': 3500, 'RM_08': 100, 'RM_09': 700,
    'RM_11': 9000, 'RM_12': 5000, 'RM_13': 5000, 'RM_17': 700, 'RM_18': 800,
    'RM_20': 6000, 'RM_23': 1100, 'RM_04': 2100
}

# Initialize Parameters for Post Mixer Costs
model.postmixer_costs = Param(model.RM, initialize=post_mixer_costs_dict, default=0)

# Define the Mixer Total SUB
def mixer_SUB (model,sub):
    return sum (model.Mixer_RM_qty[rm] * model.composition[sub, rm] for rm in model.RM)

# Define the Post Mixer Total SUB
def prod_A_SUB (model,sub):
    return sum (model.prod_A[rm] * model.composition[sub, rm] for rm in model.RM)
def prod_B_SUB (model,sub):
    return sum (model.prod_B[rm] * model.composition[sub, rm] for rm in model.RM)
def prod_C_SUB (model,sub):
    return sum (model.prod_C[rm] * model.composition[sub, rm] for rm in model.RM)
def prod_D_SUB (model,sub):
    return sum (model.prod_D[rm] * model.composition[sub, rm] for rm in model.RM)
def prod_E_SUB (model,sub):
    return sum (model.prod_E[rm] * model.composition[sub, rm] for rm in model.RM)

### MODEL EXPRESSION AND CONSTRAINTS IN MIXER ###
# Define model_5 as a Mixer-Related Equation
def model_5(model):
    mixer_SUB_02 = mixer_SUB(model, 'SUB_02')
    mixer_SUB_03 = mixer_SUB(model, 'SUB_03')
    mixer_SUB_07 = mixer_SUB(model, 'SUB_07')
    mixer_SUB_26 = mixer_SUB(model, 'SUB_26')
    mixer_SUB_10 = mixer_SUB(model, 'SUB_10')
    mixer_SUB_09 = mixer_SUB(model, 'SUB_09')
    mixer_SUB_15 = mixer_SUB(model, 'SUB_15')
    mixer_SUB_14 = mixer_SUB(model, 'SUB_14')
    mixer_SUB_23 = mixer_SUB(model, 'SUB_23')
    mixer_SUB_13 = mixer_SUB(model, 'SUB_13')
    mixer_SUB_22 = mixer_SUB(model, 'SUB_22')
    mixer_SUB_12 = mixer_SUB(model, 'SUB_12')
    mixer_SUB_11 = mixer_SUB(model, 'SUB_11')
    mixer_SUB_26 = mixer_SUB(model, 'SUB_26')

    RM_06 = model.Mixer_RM_qty['RM_06']

    return (1 - (mixer_SUB_02 / (mixer_SUB_02 + 0.00001))) * (
        2 + 55 * (mixer_SUB_07 / 0.05 * 0.75) - 31 * mixer_SUB_03 + 172 * mixer_SUB_26 + 172 * RM_06 * 0.005
        + 69 * mixer_SUB_10 - 320 * (mixer_SUB_09 / 0.48 * 0.4794) - 9 * mixer_SUB_15 + 20 * mixer_SUB_14
        - 20 * (mixer_SUB_07 / 0.05 * 0.0018) + 57 * mixer_SUB_23 + 10 * mixer_SUB_13 - 44 * mixer_SUB_22
        - 60 * mixer_SUB_12 + 9 * mixer_SUB_11 - 618 * (mixer_SUB_07 / 0.05 * 0.75) * (mixer_SUB_07 / 0.05 * 0.75)
        - 976 * mixer_SUB_03 * mixer_SUB_03 - 2120 * mixer_SUB_26 * mixer_SUB_26 - 2120 * 2 * mixer_SUB_26 * RM_06 * 0.005
        - 2120 * RM_06 * RM_06 * 0.005 * 0.005 + 1425 * mixer_SUB_10 * mixer_SUB_10
        + 187 * (mixer_SUB_09 / 0.48 * 0.47) * (mixer_SUB_09 / 0.48 * 0.47) - 206 * mixer_SUB_15 * mixer_SUB_15
        - 339 * mixer_SUB_14 * mixer_SUB_14 + 339 * 2 * mixer_SUB_14 * (mixer_SUB_07 / 0.05 * 0.0018)
        - 339 * (mixer_SUB_07 / 0.05 * 0.0018) * (mixer_SUB_07 / 0.05 * 0.0018)
        - 287 * mixer_SUB_23 * mixer_SUB_23 + 145 * mixer_SUB_13 * mixer_SUB_13 + 12 * mixer_SUB_22 * mixer_SUB_22
        + 2145 * mixer_SUB_12 * mixer_SUB_12 + 2 * mixer_SUB_11 * mixer_SUB_11
        + 1110 * (mixer_SUB_07 / 0.05 * 0.75) * mixer_SUB_03 + 682 * (mixer_SUB_07 / 0.05 * 0.75) * mixer_SUB_26
        + 682 * (mixer_SUB_07 / 0.05 * 0.75) * RM_06 * 0.005 + 666 * (mixer_SUB_07 / 0.05 * 0.75) * mixer_SUB_10
        - 2222 * (mixer_SUB_07 / 0.05 * 0.75) * (mixer_SUB_09 / 0.48 * 0.48) - 541 * (mixer_SUB_07 / 0.05 * 0.75) * mixer_SUB_15
        + 835 * (mixer_SUB_07 / 0.05 * 0.75) * mixer_SUB_14 - 835 * (mixer_SUB_07 / 0.05 * 0.75) * (mixer_SUB_07 / 0.05 * 0.0018)
        - 215 * (mixer_SUB_07 / 0.05 * 0.75) * mixer_SUB_23 + 402 * (mixer_SUB_07 / 0.05 * 0.75) * mixer_SUB_13
        + 176 * (mixer_SUB_07 / 0.05 * 0.75) * mixer_SUB_22 + 111 * (mixer_SUB_07 / 0.05 * 0.75) * mixer_SUB_12
        - 317 * (mixer_SUB_07 / 0.05 * 0.75) * mixer_SUB_11 - 4235 * mixer_SUB_03 * mixer_SUB_26
        - 4235 * mixer_SUB_03 * RM_06 * 0.005 + 640 * mixer_SUB_03 * mixer_SUB_10
        + 2235 * mixer_SUB_03 * (mixer_SUB_09 / 0.48 * 0.48) - 2 * mixer_SUB_03 * mixer_SUB_15
        + 1191 * mixer_SUB_03 * mixer_SUB_14 - 1191 * mixer_SUB_03 * (mixer_SUB_07 / 0.05 * 0.0018)
        + 896 * mixer_SUB_03 * mixer_SUB_23 - 2136 * mixer_SUB_03 * mixer_SUB_13 - 30 * mixer_SUB_03 * mixer_SUB_22
        + 422 * mixer_SUB_03 * mixer_SUB_12 + 345 * mixer_SUB_03 * mixer_SUB_11
        - 618 * mixer_SUB_26 * mixer_SUB_10 - 618 * RM_06 * 0.005 * mixer_SUB_10
        - 6484 * mixer_SUB_26 * (mixer_SUB_09 / 0.48 * 0.48) - 6484 * RM_06 * 0.005 * (mixer_SUB_09 / 0.48 * 0.48)
        - 394 * mixer_SUB_26 * mixer_SUB_15 - 394 * RM_06 * 0.005 * mixer_SUB_15
        - 1601 * mixer_SUB_26 * mixer_SUB_14 - 1601 * RM_06 * 0.005 * mixer_SUB_14
        + 1601 * mixer_SUB_26 * (mixer_SUB_07 / 0.05 * 0.0018) + 1601 * RM_06 * 0.005 * (mixer_SUB_07 / 0.05 * 0.0018)
        - 1443 * mixer_SUB_26 * mixer_SUB_23 - 1443 * RM_06 * 0.005 * mixer_SUB_23
        + 2012 * mixer_SUB_26 * mixer_SUB_13 + 2012 * RM_06 * 0.005 * mixer_SUB_13
        + 1270 * mixer_SUB_26 * mixer_SUB_22 + 1270 * RM_06 * 0.005 * mixer_SUB_22
        + 2190 * mixer_SUB_26 * mixer_SUB_12 + 2190 * RM_06 * 0.005 * mixer_SUB_12
        + 614 * mixer_SUB_26 * mixer_SUB_11 + 614 * RM_06 * 0.005 * mixer_SUB_11
        + 274 * mixer_SUB_10 * (mixer_SUB_09 / 0.48 * 0.48) + 145 * mixer_SUB_10 * mixer_SUB_15
        + 703 * mixer_SUB_10 * mixer_SUB_14 - 703 * mixer_SUB_10 * (mixer_SUB_07 / 0.05 * 0.0018)
        - 218 * mixer_SUB_10 * mixer_SUB_23 - 3414 * mixer_SUB_10 * mixer_SUB_13
        - 558 * mixer_SUB_10 * mixer_SUB_22 - 784 * mixer_SUB_10 * mixer_SUB_12
        - 428 * mixer_SUB_10 * mixer_SUB_11 + 178 * (mixer_SUB_09 / 0.48 * 0.48) * mixer_SUB_15
        + 3245 * (mixer_SUB_09 / 0.48 * 0.48) * mixer_SUB_14 - 3245 * (mixer_SUB_09 / 0.48 * 0.48) * (mixer_SUB_07 / 0.05 * 0.0018)
        + 920 * (mixer_SUB_09 / 0.48 * 0.48) * mixer_SUB_23 - 5576 * (mixer_SUB_09 / 0.48 * 0.48) * mixer_SUB_13
        - 1995 * (mixer_SUB_09 / 0.48 * 0.48) * mixer_SUB_22 + 5996 * (mixer_SUB_09 / 0.48 * 0.48) * mixer_SUB_12
        + 1890 * (mixer_SUB_09 / 0.48 * 0.48) * mixer_SUB_11 + 334 * mixer_SUB_15 * mixer_SUB_14
        - 334 * mixer_SUB_15 * (mixer_SUB_07 / 0.05 * 0.0018) - 94 * mixer_SUB_15 * mixer_SUB_23
        + 1306 * mixer_SUB_15 * mixer_SUB_13 + 426 * mixer_SUB_15 * mixer_SUB_22
        + 521 * mixer_SUB_15 * mixer_SUB_12 - 105 * mixer_SUB_15 * mixer_SUB_11
        - 5 * mixer_SUB_14 * mixer_SUB_23 + 5 * (mixer_SUB_07 / 0.05 * 0.0018) * mixer_SUB_23
        - 3328 * mixer_SUB_14 * mixer_SUB_13 - 3328 * (mixer_SUB_07 / 0.05 * 0.0018) * mixer_SUB_13
        + 937 * mixer_SUB_14 * mixer_SUB_22 - 937 * (mixer_SUB_07 / 0.05 * 0.0018) * mixer_SUB_22
        - 1170 * mixer_SUB_14 * mixer_SUB_12 + 1170 * (mixer_SUB_07 / 0.05 * 0.0018) * mixer_SUB_12
        - 94 * mixer_SUB_14 * mixer_SUB_11 + 94 * (mixer_SUB_07 / 0.05 * 0.0018) * mixer_SUB_11
        + 357 * mixer_SUB_23 * mixer_SUB_13 + 303 * mixer_SUB_23 * mixer_SUB_22
        + 552 * mixer_SUB_23 * mixer_SUB_12 - 206 * mixer_SUB_23 * mixer_SUB_11
        + 173 * mixer_SUB_13 * mixer_SUB_22 - 4444 * mixer_SUB_13 * mixer_SUB_12
        + 996 * mixer_SUB_13 * mixer_SUB_11 - 985 * mixer_SUB_22 * mixer_SUB_12
        + 122 * mixer_SUB_22 * mixer_SUB_11 - 65 * mixer_SUB_12 * mixer_SUB_11) + 10 * (mixer_SUB_02 / (mixer_SUB_02 + 0.00001))

# model_5 constraint for the Mixer Component
model.model_5_constraint = Constraint(expr=model_5(model) >= 9)

# Define model_6 as an expression (Mixer Cost)
def model_6(model):
    return sum(model.mixercosts[rm] * model.Mixer_RM_qty[rm] for rm in model.RM)

# Total Mixer Composition Constraints
def mixer_total_quantity_rule(model):
    return sum(model.Mixer_RM_qty[rm] for rm in model.RM) == 1.0
model.mixer_total_quantity_constraint = Constraint(rule=mixer_total_quantity_rule)

# Mixer Bound Constraint
model.lower_bound_RM_15 = Constraint(expr=model.Mixer_RM_qty['RM_15'] >= 0.005)
model.lower_bound_RM_14 = Constraint(expr=model.Mixer_RM_qty['RM_14'] >= 0.06)
model.lower_bound_RM_05 = Constraint(expr=model.Mixer_RM_qty['RM_05'] >= 0.25)
model.lower_bound_RM_18 = Constraint(expr=model.Mixer_RM_qty['RM_18'] >= 0.01)

# Component Constraint in the Mixer
model.mixer_component_RM_03 = Constraint(expr=model.Mixer_RM_qty['RM_03'] == 0)
model.mixer_component_RM_04 = Constraint(expr=model.Mixer_RM_qty['RM_04'] == 0)
model.mixer_component_RM_07 = Constraint(expr=model.Mixer_RM_qty['RM_07'] == 0)
model.mixer_component_RM_08 = Constraint(expr=model.Mixer_RM_qty['RM_08'] == 0)
model.mixer_component_RM_09 = Constraint(expr=model.Mixer_RM_qty['RM_09'] == 0)
model.mixer_component_RM_11 = Constraint(expr=model.Mixer_RM_qty['RM_11'] == 0)
model.mixer_component_RM_12 = Constraint(expr=model.Mixer_RM_qty['RM_12'] == 0)
model.mixer_component_RM_13 = Constraint(expr=model.Mixer_RM_qty['RM_13'] == 0)
model.mixer_component_RM_17 = Constraint(expr=model.Mixer_RM_qty['RM_17'] == 0)
model.mixer_component_RM_20 = Constraint(expr=model.Mixer_RM_qty['RM_20'] == 0)
model.mixer_component_RM_23 = Constraint(expr=model.Mixer_RM_qty['RM_23'] == 0)

### MODEL EXPRESSIONS PROD A ###

# Define model_1A as an expression (SRI Model)
def model_1A(model):
    Total_Prod_A_SUB_01 = model.Mixer_Level_A * mixer_SUB(model, 'SUB_01') + prod_A_SUB(model,'SUB_01')
    Total_Prod_A_SUB_02 = model.Mixer_Level_A * mixer_SUB(model, 'SUB_02') + prod_A_SUB(model,'SUB_02')
    Total_Prod_A_SUB_09 = model.Mixer_Level_A * mixer_SUB(model, 'SUB_09') + prod_A_SUB(model,'SUB_09')
    Total_Prod_A_SUB_11 = model.Mixer_Level_A * mixer_SUB(model, 'SUB_11') + prod_A_SUB(model,'SUB_11')
    Total_Prod_A_SUB_18 = model.Mixer_Level_A * mixer_SUB(model, 'SUB_18') + prod_A_SUB(model,'SUB_18')
    Total_Prod_A_SUB_19 = model.Mixer_Level_A * mixer_SUB(model, 'SUB_19') + prod_A_SUB(model,'SUB_19')
    Total_Prod_A_SUB_27 = model.Mixer_Level_A * mixer_SUB(model, 'SUB_27') + prod_A_SUB(model,'SUB_27')

    return (
        53 - 6 * Total_Prod_A_SUB_11 - 5 * Total_Prod_A_SUB_02 + 1172 * Total_Prod_A_SUB_01 - 400 * Total_Prod_A_SUB_09 + 1874 * Total_Prod_A_SUB_18 +
        23 * Total_Prod_A_SUB_27 + 183510 * Total_Prod_A_SUB_19 - 2038 * Total_Prod_A_SUB_27 * Total_Prod_A_SUB_27 - 199367 * Total_Prod_A_SUB_18 * Total_Prod_A_SUB_18 +
        2561 * Total_Prod_A_SUB_27 * Total_Prod_A_SUB_09 + 130 * Total_Prod_A_SUB_11 * Total_Prod_A_SUB_09 + 2592 * Total_Prod_A_SUB_09 * Total_Prod_A_SUB_09 -
        103429177 * Total_Prod_A_SUB_19 * Total_Prod_A_SUB_19 - 1048878 * Total_Prod_A_SUB_19 * Total_Prod_A_SUB_18
    )

model.model_1A = Expression(rule=model_1A)

# Define model_2A as an expression (SRI Model)
def model_2A(model):
    Total_Prod_A_SUB_02 = model.Mixer_Level_A * mixer_SUB(model, 'SUB_02') + prod_A_SUB(model,'SUB_02')
    Total_Prod_A_SUB_09 = model.Mixer_Level_A * mixer_SUB(model, 'SUB_09') + prod_A_SUB(model,'SUB_09')
    Total_Prod_A_SUB_11 = model.Mixer_Level_A * mixer_SUB(model, 'SUB_11') + prod_A_SUB(model,'SUB_11')
    Total_Prod_A_SUB_15 = model.Mixer_Level_A * mixer_SUB(model, 'SUB_15') + prod_A_SUB(model,'SUB_15')
    Total_Prod_A_SUB_16 = model.Mixer_Level_A * mixer_SUB(model, 'SUB_16') + prod_A_SUB(model,'SUB_16')
    Total_Prod_A_SUB_21 = model.Mixer_Level_A * mixer_SUB(model, 'SUB_21') + prod_A_SUB(model,'SUB_21')
    Total_Prod_A_SUB_23 = model.Mixer_Level_A * mixer_SUB(model, 'SUB_23') + prod_A_SUB(model,'SUB_23')
    Total_Prod_A_SUB_24 = model.Mixer_Level_A * mixer_SUB(model, 'SUB_24') + prod_A_SUB(model,'SUB_24')
    Total_Prod_A_SUB_26 = model.Mixer_Level_A * mixer_SUB(model, 'SUB_26') + prod_A_SUB(model,'SUB_26')
    Total_Prod_A_SUB_27 = model.Mixer_Level_A * mixer_SUB(model, 'SUB_27') + prod_A_SUB(model,'SUB_27')

    return (
        29 + 51 * Total_Prod_A_SUB_21 + 42 * Total_Prod_A_SUB_15 + 15 * Total_Prod_A_SUB_23 + 25 * Total_Prod_A_SUB_11 + 51 * Total_Prod_A_SUB_02 + 121 * Total_Prod_A_SUB_16 -
        139 * Total_Prod_A_SUB_24 - 30 * Total_Prod_A_SUB_26 - 80 * Total_Prod_A_SUB_09 + 231 * Total_Prod_A_SUB_27 - 23 * Total_Prod_A_SUB_21 * Total_Prod_A_SUB_21 -
        36 * Total_Prod_A_SUB_21 * Total_Prod_A_SUB_15 - 27 * Total_Prod_A_SUB_21 * Total_Prod_A_SUB_23 - 21 * Total_Prod_A_SUB_21 * Total_Prod_A_SUB_02 -
        37 * Total_Prod_A_SUB_15 * Total_Prod_A_SUB_02 - 36 * Total_Prod_A_SUB_23 * Total_Prod_A_SUB_23 - 14 * Total_Prod_A_SUB_23 * Total_Prod_A_SUB_11 -
        1602 * Total_Prod_A_SUB_16 * Total_Prod_A_SUB_16 + 29 * Total_Prod_A_SUB_21 * Total_Prod_A_SUB_24 - 5 * Total_Prod_A_SUB_21 * Total_Prod_A_SUB_11 +
        178 * Total_Prod_A_SUB_24 * Total_Prod_A_SUB_16 + 54 * Total_Prod_A_SUB_24 * Total_Prod_A_SUB_26 - 25 * Total_Prod_A_SUB_15 * Total_Prod_A_SUB_11 -
        2912 * Total_Prod_A_SUB_27 * Total_Prod_A_SUB_27 + 68 * Total_Prod_A_SUB_27 * Total_Prod_A_SUB_11 - 20 * Total_Prod_A_SUB_11 * Total_Prod_A_SUB_02 +
        85 * Total_Prod_A_SUB_24 * Total_Prod_A_SUB_27 - 16 * Total_Prod_A_SUB_15 * Total_Prod_A_SUB_15 + 86 * Total_Prod_A_SUB_27 * Total_Prod_A_SUB_23 -
        50 * Total_Prod_A_SUB_27 * Total_Prod_A_SUB_02 + 222 * Total_Prod_A_SUB_27 * Total_Prod_A_SUB_26 + 1447 * Total_Prod_A_SUB_27 * Total_Prod_A_SUB_09 -
        57 * Total_Prod_A_SUB_11 * Total_Prod_A_SUB_16 + 37 * Total_Prod_A_SUB_11 * Total_Prod_A_SUB_26
    )

model.model_2A = Expression(rule=model_2A)

# Define model_3A as an expression (SRI Model)
def model_3A(model):
    Total_Prod_A_SUB_01 = model.Mixer_Level_A * mixer_SUB(model, 'SUB_01') + prod_A_SUB(model,'SUB_01')
    Total_Prod_A_SUB_02 = model.Mixer_Level_A * mixer_SUB(model, 'SUB_02') + prod_A_SUB(model,'SUB_02')
    Total_Prod_A_SUB_03 = model.Mixer_Level_A * mixer_SUB(model, 'SUB_03') + prod_A_SUB(model,'SUB_03')
    Total_Prod_A_SUB_11 = model.Mixer_Level_A * mixer_SUB(model, 'SUB_11') + prod_A_SUB(model,'SUB_11')
    Total_Prod_A_SUB_15 = model.Mixer_Level_A * mixer_SUB(model, 'SUB_15') + prod_A_SUB(model,'SUB_15')
    Total_Prod_A_SUB_18 = model.Mixer_Level_A * mixer_SUB(model, 'SUB_18') + prod_A_SUB(model,'SUB_18')
    Total_Prod_A_SUB_21 = model.Mixer_Level_A * mixer_SUB(model, 'SUB_21') + prod_A_SUB(model,'SUB_21')
    Total_Prod_A_SUB_23 = model.Mixer_Level_A * mixer_SUB(model, 'SUB_23') + prod_A_SUB(model,'SUB_23')
    Total_Prod_A_SUB_26 = model.Mixer_Level_A * mixer_SUB(model, 'SUB_26') + prod_A_SUB(model,'SUB_26')

    return (
        9 + 12 * Total_Prod_A_SUB_21 - 0.5 * Total_Prod_A_SUB_15 + 10 * Total_Prod_A_SUB_23 + 7 * Total_Prod_A_SUB_11 + 12 * Total_Prod_A_SUB_02 + 34 * Total_Prod_A_SUB_03 -
        13976 * Total_Prod_A_SUB_01 + 9 * Total_Prod_A_SUB_26 + 4069 * Total_Prod_A_SUB_18 - 8 * Total_Prod_A_SUB_21 * Total_Prod_A_SUB_21 - 2 * Total_Prod_A_SUB_21 * Total_Prod_A_SUB_15 -
        23 * Total_Prod_A_SUB_21 * Total_Prod_A_SUB_23 - 8 * Total_Prod_A_SUB_21 * Total_Prod_A_SUB_02 - 7 * Total_Prod_A_SUB_15 * Total_Prod_A_SUB_02 + 1754 * Total_Prod_A_SUB_15 * Total_Prod_A_SUB_18 +
        0.4 * Total_Prod_A_SUB_23 * Total_Prod_A_SUB_03 + 120 * Total_Prod_A_SUB_26 * Total_Prod_A_SUB_26 + 17776885 * Total_Prod_A_SUB_01 * Total_Prod_A_SUB_01 - 7 * Total_Prod_A_SUB_11 * Total_Prod_A_SUB_02 -
        364999 * Total_Prod_A_SUB_18 * Total_Prod_A_SUB_18 - 564 * Total_Prod_A_SUB_03 * Total_Prod_A_SUB_03
    )

model.model_3A = Expression(rule=model_3A)

# Define model_4A as a constraint based on model_1A, model_2A, and model_3A
def model_4A(model):
    return 0.3 * model.model_1A + 0.2 * model.model_2A + 0.5 * model.model_3A

# Define model_7A as an expression (Post Mixer Cost)
def model_7A(model):
    return sum(model.postmixer_costs[rm] * model.prod_A[rm] for rm in model.RM)

### MODEL CONSTRAINTS PROD A ###

# Total PROD A Composition Constraints
def Prod_A_total_quantity_rule(model):
    return sum(model.prod_A[rm] for rm in model.RM) + model.Mixer_Level_A == 1.0
model.Prod_A_total_quantity_constraint = Constraint(rule=Prod_A_total_quantity_rule)

# model_4A constraint for the SRI
model.model_4A_constraint = Constraint(expr=model_4A(model) >= 52)

# Component Constraint in the Post Mixer
model.Prod_A_RM_01 = Constraint(expr=model.prod_A['RM_01'] == 0)
model.Prod_A_RM_02 = Constraint(expr=model.prod_A['RM_02'] == 0)
model.Prod_A_RM_05 = Constraint(expr=model.prod_A['RM_05'] == 0)
model.Prod_A_RM_06 = Constraint(expr=model.prod_A['RM_06'] == 0)
model.Prod_A_RM_10 = Constraint(expr=model.prod_A['RM_10'] == 0)
model.Prod_A_RM_14 = Constraint(expr=model.prod_A['RM_14'] == 0)
model.Prod_A_RM_15 = Constraint(expr=model.prod_A['RM_15'] == 0)
model.Prod_A_RM_16 = Constraint(expr=model.prod_A['RM_16'] == 0)
model.Prod_A_RM_18 = Constraint(expr=model.prod_A['RM_18'] == 0)
model.Prod_A_RM_19 = Constraint(expr=model.prod_A['RM_19'] == 0)
model.Prod_A_RM_21 = Constraint(expr=model.prod_A['RM_21'] == 0)
model.Prod_A_RM_22 = Constraint(expr=model.prod_A['RM_22'] == 0)

# Final Products RM Component Bounds
model.prod_A_lb_RM_12 = Constraint(expr=model.Mixer_Level_A *model.Mixer_RM_qty['RM_12'] + model.prod_A['RM_12'] >= 0.008)
model.prod_A_ub_RM_18 = Constraint(expr=model.Mixer_Level_A *model.Mixer_RM_qty['RM_18'] + model.prod_A['RM_18'] <= 0.02)
model.prod_A_ub_RM_08 = Constraint(expr=model.Mixer_Level_A *model.Mixer_RM_qty['RM_08'] + model.prod_A['RM_08'] <= 0.46)

# Final Products SUB Component Bounds
model.prod_A_lb_SUB_21 = Constraint(expr=model.Mixer_Level_A * mixer_SUB(model, 'SUB_21') + prod_A_SUB(model,'SUB_21') >= 0)
model.prod_A_ub_SUB_21 = Constraint(expr=model.Mixer_Level_A * mixer_SUB(model, 'SUB_21') + prod_A_SUB(model,'SUB_21') <= 0.75)
model.prod_A_lb_SUB_27 = Constraint(expr=model.Mixer_Level_A * mixer_SUB(model, 'SUB_27') + prod_A_SUB(model,'SUB_27') >= 0.003)
model.prod_A_ub_SUB_27 = Constraint(expr=model.Mixer_Level_A * mixer_SUB(model, 'SUB_27') + prod_A_SUB(model,'SUB_27') <= 0.02)
model.prod_A_ub_SUB_03 = Constraint(expr=model.Mixer_Level_A * mixer_SUB(model, 'SUB_03') + prod_A_SUB(model,'SUB_03') <= 0.09)
model.prod_A_lb_SUB_26 = Constraint(expr=model.Mixer_Level_A * mixer_SUB(model, 'SUB_26') + prod_A_SUB(model,'SUB_26') >= 0.02)
model.prod_A_ub_SUB_26 = Constraint(expr=model.Mixer_Level_A * mixer_SUB(model, 'SUB_26') + prod_A_SUB(model,'SUB_26') <= 0.2)
model.prod_A_lb_SUB_20 = Constraint(expr=model.Mixer_Level_A * mixer_SUB(model, 'SUB_20') + prod_A_SUB(model,'SUB_20') >= 0.001)
model.prod_A_lb_SUB_19 = Constraint(expr=model.Mixer_Level_A * mixer_SUB(model, 'SUB_19') + prod_A_SUB(model,'SUB_19') >= 0.0001)
model.prod_A_ub_SUB_19 = Constraint(expr=model.Mixer_Level_A * mixer_SUB(model, 'SUB_19') + prod_A_SUB(model,'SUB_19') <= 0.0009)
model.prod_A_lb_SUB_18 = Constraint(expr=model.Mixer_Level_A * mixer_SUB(model, 'SUB_18') + prod_A_SUB(model,'SUB_18') >= 0.0001)
model.prod_A_ub_SUB_18 = Constraint(expr=model.Mixer_Level_A * mixer_SUB(model, 'SUB_18') + prod_A_SUB(model,'SUB_18') <= 0.0065)
model.prod_A_ub_SUB_10 = Constraint(expr=model.Mixer_Level_A * mixer_SUB(model, 'SUB_10') + prod_A_SUB(model,'SUB_10') <= 0.029)
model.prod_A_lb_SUB_24 = Constraint(expr=model.Mixer_Level_A * mixer_SUB(model, 'SUB_24') + prod_A_SUB(model,'SUB_24') >= 0)
model.prod_A_ub_SUB_24 = Constraint(expr=model.Mixer_Level_A * mixer_SUB(model, 'SUB_24') + prod_A_SUB(model,'SUB_24') <= 0.2)
model.prod_A_lb_SUB_09 = Constraint(expr=model.Mixer_Level_A * mixer_SUB(model, 'SUB_09') + prod_A_SUB(model,'SUB_09') >= 0.002)
model.prod_A_ub_SUB_09 = Constraint(expr=model.Mixer_Level_A * mixer_SUB(model, 'SUB_09') + prod_A_SUB(model,'SUB_09') <= 0.05)
model.prod_A_lb_SUB_16 = Constraint(expr=model.Mixer_Level_A * mixer_SUB(model, 'SUB_16') + prod_A_SUB(model,'SUB_16') >= 0.0055)
model.prod_A_ub_SUB_16 = Constraint(expr=model.Mixer_Level_A * mixer_SUB(model, 'SUB_16') + prod_A_SUB(model,'SUB_16') <= 0.018)
model.prod_A_ub_SUB_15 = Constraint(expr=model.Mixer_Level_A * mixer_SUB(model, 'SUB_15') + prod_A_SUB(model,'SUB_15') <= 0.7)
model.prod_A_lb_SUB_23 = Constraint(expr=model.Mixer_Level_A * mixer_SUB(model, 'SUB_23') + prod_A_SUB(model,'SUB_23') >= 0)
model.prod_A_ub_SUB_23 = Constraint(expr=model.Mixer_Level_A * mixer_SUB(model, 'SUB_23') + prod_A_SUB(model,'SUB_23') <= 0.16)

### MODEL EXPRESSION PROD B ###

# Define model_1B as an expression (SRI Model)
def model_1B(model):
    Total_Prod_B_SUB_01 = model.Mixer_Level_B * mixer_SUB(model, 'SUB_01') + prod_B_SUB(model,'SUB_01')
    Total_Prod_B_SUB_02 = model.Mixer_Level_B * mixer_SUB(model, 'SUB_02') + prod_B_SUB(model,'SUB_02')
    Total_Prod_B_SUB_09 = model.Mixer_Level_B * mixer_SUB(model, 'SUB_09') + prod_B_SUB(model,'SUB_09')
    Total_Prod_B_SUB_11 = model.Mixer_Level_B * mixer_SUB(model, 'SUB_11') + prod_B_SUB(model,'SUB_11')
    Total_Prod_B_SUB_18 = model.Mixer_Level_B * mixer_SUB(model, 'SUB_18') + prod_B_SUB(model,'SUB_18')
    Total_Prod_B_SUB_19 = model.Mixer_Level_B * mixer_SUB(model, 'SUB_19') + prod_B_SUB(model,'SUB_19')
    Total_Prod_B_SUB_27 = model.Mixer_Level_B * mixer_SUB(model, 'SUB_27') + prod_B_SUB(model,'SUB_27')

    return (
        53 - 6 * Total_Prod_B_SUB_11 - 5 * Total_Prod_B_SUB_02 + 1172 * Total_Prod_B_SUB_01 - 400 * Total_Prod_B_SUB_09 + 1874 * Total_Prod_B_SUB_18 +
        23 * Total_Prod_B_SUB_27 + 183510 * Total_Prod_B_SUB_19 - 2038 * Total_Prod_B_SUB_27 * Total_Prod_B_SUB_27 - 199367 * Total_Prod_B_SUB_18 * Total_Prod_B_SUB_18 +
        2561 * Total_Prod_B_SUB_27 * Total_Prod_B_SUB_09 + 130 * Total_Prod_B_SUB_11 * Total_Prod_B_SUB_09 + 2592 * Total_Prod_B_SUB_09 * Total_Prod_B_SUB_09 -
        103429177 * Total_Prod_B_SUB_19 * Total_Prod_B_SUB_19 - 1048878 * Total_Prod_B_SUB_19 * Total_Prod_B_SUB_18
    )

model.model_1B = Expression(rule=model_1B)

# Define model_2B as an expression (SRI Model)
def model_2B(model):
    Total_Prod_B_SUB_02 = model.Mixer_Level_B * mixer_SUB(model, 'SUB_02') + prod_B_SUB(model,'SUB_02')
    Total_Prod_B_SUB_09 = model.Mixer_Level_B * mixer_SUB(model, 'SUB_09') + prod_B_SUB(model,'SUB_09')
    Total_Prod_B_SUB_11 = model.Mixer_Level_B * mixer_SUB(model, 'SUB_11') + prod_B_SUB(model,'SUB_11')
    Total_Prod_B_SUB_15 = model.Mixer_Level_B * mixer_SUB(model, 'SUB_15') + prod_B_SUB(model,'SUB_15')
    Total_Prod_B_SUB_16 = model.Mixer_Level_B * mixer_SUB(model, 'SUB_16') + prod_B_SUB(model,'SUB_16')
    Total_Prod_B_SUB_21 = model.Mixer_Level_B * mixer_SUB(model, 'SUB_21') + prod_B_SUB(model,'SUB_21')
    Total_Prod_B_SUB_23 = model.Mixer_Level_B * mixer_SUB(model, 'SUB_23') + prod_B_SUB(model,'SUB_23')
    Total_Prod_B_SUB_24 = model.Mixer_Level_B * mixer_SUB(model, 'SUB_24') + prod_B_SUB(model,'SUB_24')
    Total_Prod_B_SUB_26 = model.Mixer_Level_B * mixer_SUB(model, 'SUB_26') + prod_B_SUB(model,'SUB_26')
    Total_Prod_B_SUB_27 = model.Mixer_Level_B * mixer_SUB(model, 'SUB_27') + prod_B_SUB(model,'SUB_27')

    return (
        29 + 51 * Total_Prod_B_SUB_21 + 42 * Total_Prod_B_SUB_15 + 15 * Total_Prod_B_SUB_23 + 25 * Total_Prod_B_SUB_11 + 51 * Total_Prod_B_SUB_02 + 121 * Total_Prod_B_SUB_16 -
        139 * Total_Prod_B_SUB_24 - 30 * Total_Prod_B_SUB_26 - 80 * Total_Prod_B_SUB_09 + 231 * Total_Prod_B_SUB_27 - 23 * Total_Prod_B_SUB_21 * Total_Prod_B_SUB_21 -
        36 * Total_Prod_B_SUB_21 * Total_Prod_B_SUB_15 - 27 * Total_Prod_B_SUB_21 * Total_Prod_B_SUB_23 - 21 * Total_Prod_B_SUB_21 * Total_Prod_B_SUB_02 -
        37 * Total_Prod_B_SUB_15 * Total_Prod_B_SUB_02 - 36 * Total_Prod_B_SUB_23 * Total_Prod_B_SUB_23 - 14 * Total_Prod_B_SUB_23 * Total_Prod_B_SUB_11 -
        1602 * Total_Prod_B_SUB_16 * Total_Prod_B_SUB_16 + 29 * Total_Prod_B_SUB_21 * Total_Prod_B_SUB_24 - 5 * Total_Prod_B_SUB_21 * Total_Prod_B_SUB_11 +
        178 * Total_Prod_B_SUB_24 * Total_Prod_B_SUB_16 + 54 * Total_Prod_B_SUB_24 * Total_Prod_B_SUB_26 - 25 * Total_Prod_B_SUB_15 * Total_Prod_B_SUB_11 -
        2912 * Total_Prod_B_SUB_27 * Total_Prod_B_SUB_27 + 68 * Total_Prod_B_SUB_27 * Total_Prod_B_SUB_11 - 20 * Total_Prod_B_SUB_11 * Total_Prod_B_SUB_02 +
        85 * Total_Prod_B_SUB_24 * Total_Prod_B_SUB_27 - 16 * Total_Prod_B_SUB_15 * Total_Prod_B_SUB_15 + 86 * Total_Prod_B_SUB_27 * Total_Prod_B_SUB_23 -
        50 * Total_Prod_B_SUB_27 * Total_Prod_B_SUB_02 + 222 * Total_Prod_B_SUB_27 * Total_Prod_B_SUB_26 + 1447 * Total_Prod_B_SUB_27 * Total_Prod_B_SUB_09 -
        57 * Total_Prod_B_SUB_11 * Total_Prod_B_SUB_16 + 37 * Total_Prod_B_SUB_11 * Total_Prod_B_SUB_26
    )

model.model_2B = Expression(rule=model_2B)

# Define model_3B as an expression (SRI Model)
def model_3B(model):
    Total_Prod_B_SUB_01 = model.Mixer_Level_B * mixer_SUB(model, 'SUB_01') + prod_B_SUB(model,'SUB_01')
    Total_Prod_B_SUB_02 = model.Mixer_Level_B * mixer_SUB(model, 'SUB_02') + prod_B_SUB(model,'SUB_02')
    Total_Prod_B_SUB_03 = model.Mixer_Level_B * mixer_SUB(model, 'SUB_03') + prod_B_SUB(model,'SUB_03')
    Total_Prod_B_SUB_11 = model.Mixer_Level_B * mixer_SUB(model, 'SUB_11') + prod_B_SUB(model,'SUB_11')
    Total_Prod_B_SUB_15 = model.Mixer_Level_B * mixer_SUB(model, 'SUB_15') + prod_B_SUB(model,'SUB_15')
    Total_Prod_B_SUB_18 = model.Mixer_Level_B * mixer_SUB(model, 'SUB_18') + prod_B_SUB(model,'SUB_18')
    Total_Prod_B_SUB_21 = model.Mixer_Level_B * mixer_SUB(model, 'SUB_21') + prod_B_SUB(model,'SUB_21')
    Total_Prod_B_SUB_23 = model.Mixer_Level_B * mixer_SUB(model, 'SUB_23') + prod_B_SUB(model,'SUB_23')
    Total_Prod_B_SUB_26 = model.Mixer_Level_B * mixer_SUB(model, 'SUB_26') + prod_B_SUB(model,'SUB_26')

    return (
        9 + 12 * Total_Prod_B_SUB_21 - 0.5 * Total_Prod_B_SUB_15 + 10 * Total_Prod_B_SUB_23 + 7 * Total_Prod_B_SUB_11 + 12 * Total_Prod_B_SUB_02 + 34 * Total_Prod_B_SUB_03 -
        13976 * Total_Prod_B_SUB_01 + 9 * Total_Prod_B_SUB_26 + 4069 * Total_Prod_B_SUB_18 - 8 * Total_Prod_B_SUB_21 * Total_Prod_B_SUB_21 - 2 * Total_Prod_B_SUB_21 * Total_Prod_B_SUB_15 -
        23 * Total_Prod_B_SUB_21 * Total_Prod_B_SUB_23 - 8 * Total_Prod_B_SUB_21 * Total_Prod_B_SUB_02 - 7 * Total_Prod_B_SUB_15 * Total_Prod_B_SUB_02 + 1754 * Total_Prod_B_SUB_15 * Total_Prod_B_SUB_18 +
        0.4 * Total_Prod_B_SUB_23 * Total_Prod_B_SUB_03 + 120 * Total_Prod_B_SUB_26 * Total_Prod_B_SUB_26 + 17776885 * Total_Prod_B_SUB_01 * Total_Prod_B_SUB_01 - 7 * Total_Prod_B_SUB_11 * Total_Prod_B_SUB_02 -
        364999 * Total_Prod_B_SUB_18 * Total_Prod_B_SUB_18 - 564 * Total_Prod_B_SUB_03 * Total_Prod_B_SUB_03
    )

model.model_3B = Expression(rule=model_3B)

# Define model_4B as a constraint based on model_1B, model_2B, and model_3B
def model_4B(model):
    return 0.3 * model.model_1B + 0.2 * model.model_2B + 0.5 * model.model_3B

# Define model_7B as an expression (Post Mixer Cost)
def model_7B(model):
    return sum(model.postmixer_costs[rm] * model.prod_B[rm] for rm in model.RM)


### MODEL CONSTRAINTS PROD B ###

# Total Post Mixer Composition Constraints
def Prod_B_total_quantity_rule(model):
    return sum(model.prod_B[rm] for rm in model.RM) + model.Mixer_Level_B == 1.0
model.Prod_B_total_quantity_constraint = Constraint(rule=Prod_B_total_quantity_rule)

# model_4B constraint for the SRI
model.model_4B_constraint = Constraint(expr=model_4B(model) >= 47)

# Component Constraint in the Post Mixer
model.Prod_B_RM_01 = Constraint(expr=model.prod_B['RM_01'] == 0)
model.Prod_B_RM_02 = Constraint(expr=model.prod_B['RM_02'] == 0)
model.Prod_B_RM_05 = Constraint(expr=model.prod_B['RM_05'] == 0)
model.Prod_B_RM_06 = Constraint(expr=model.prod_B['RM_06'] == 0)
model.Prod_B_RM_10 = Constraint(expr=model.prod_B['RM_10'] == 0)
model.Prod_B_RM_14 = Constraint(expr=model.prod_B['RM_14'] == 0)
model.Prod_B_RM_15 = Constraint(expr=model.prod_B['RM_15'] == 0)
model.Prod_B_RM_16 = Constraint(expr=model.prod_B['RM_16'] == 0)
model.Prod_B_RM_18 = Constraint(expr=model.prod_B['RM_18'] == 0)
model.Prod_B_RM_19 = Constraint(expr=model.prod_B['RM_19'] == 0)
model.Prod_B_RM_21 = Constraint(expr=model.prod_B['RM_21'] == 0)
model.Prod_B_RM_22 = Constraint(expr=model.prod_B['RM_22'] == 0)

# Final Products RM Component Bounds
model.prod_B_lb_RM_12 = Constraint(expr=model.Mixer_Level_B *model.Mixer_RM_qty['RM_12'] + model.prod_B['RM_12'] >= 0.008)
model.prod_B_ub_RM_18 = Constraint(expr=model.Mixer_Level_B *model.Mixer_RM_qty['RM_18'] + model.prod_B['RM_18'] <= 0.02)
model.prod_B_ub_RM_08 = Constraint(expr=model.Mixer_Level_B *model.Mixer_RM_qty['RM_08'] + model.prod_B['RM_08'] <= 0.46)

# Final Products SUB Component Bounds
model.prod_B_lb_SUB_21 = Constraint(expr=model.Mixer_Level_B * mixer_SUB(model, 'SUB_21') + prod_B_SUB(model,'SUB_21') >= 0)
model.prod_B_ub_SUB_21 = Constraint(expr=model.Mixer_Level_B * mixer_SUB(model, 'SUB_21') + prod_B_SUB(model,'SUB_21') <= 0.75)
model.prod_B_lb_SUB_27 = Constraint(expr=model.Mixer_Level_B * mixer_SUB(model, 'SUB_27') + prod_B_SUB(model,'SUB_27') >= 0.003)
model.prod_B_ub_SUB_27 = Constraint(expr=model.Mixer_Level_B * mixer_SUB(model, 'SUB_27') + prod_B_SUB(model,'SUB_27') <= 0.02)
model.prod_B_ub_SUB_03 = Constraint(expr=model.Mixer_Level_B * mixer_SUB(model, 'SUB_03') + prod_B_SUB(model,'SUB_03') <= 0.09)
model.prod_B_lb_SUB_26 = Constraint(expr=model.Mixer_Level_B * mixer_SUB(model, 'SUB_26') + prod_B_SUB(model,'SUB_26') >= 0.02)
model.prod_B_ub_SUB_26 = Constraint(expr=model.Mixer_Level_B * mixer_SUB(model, 'SUB_26') + prod_B_SUB(model,'SUB_26') <= 0.2)
model.prod_B_lb_SUB_20 = Constraint(expr=model.Mixer_Level_B * mixer_SUB(model, 'SUB_20') + prod_B_SUB(model,'SUB_20') >= 0.001)
model.prod_B_lb_SUB_19 = Constraint(expr=model.Mixer_Level_B * mixer_SUB(model, 'SUB_19') + prod_B_SUB(model,'SUB_19') >= 0.0001)
model.prod_B_ub_SUB_19 = Constraint(expr=model.Mixer_Level_B * mixer_SUB(model, 'SUB_19') + prod_B_SUB(model,'SUB_19') <= 0.0009)
model.prod_B_lb_SUB_18 = Constraint(expr=model.Mixer_Level_B * mixer_SUB(model, 'SUB_18') + prod_B_SUB(model,'SUB_18') >= 0.0001)
model.prod_B_ub_SUB_18 = Constraint(expr=model.Mixer_Level_B * mixer_SUB(model, 'SUB_18') + prod_B_SUB(model,'SUB_18') <= 0.0065)
model.prod_B_ub_SUB_10 = Constraint(expr=model.Mixer_Level_B * mixer_SUB(model, 'SUB_10') + prod_B_SUB(model,'SUB_10') <= 0.029)
model.prod_B_lb_SUB_24 = Constraint(expr=model.Mixer_Level_B * mixer_SUB(model, 'SUB_24') + prod_B_SUB(model,'SUB_24') >= 0)
model.prod_B_ub_SUB_24 = Constraint(expr=model.Mixer_Level_B * mixer_SUB(model, 'SUB_24') + prod_B_SUB(model,'SUB_24') <= 0.2)
model.prod_B_lb_SUB_09 = Constraint(expr=model.Mixer_Level_B * mixer_SUB(model, 'SUB_09') + prod_B_SUB(model,'SUB_09') >= 0.002)
model.prod_B_ub_SUB_09 = Constraint(expr=model.Mixer_Level_B * mixer_SUB(model, 'SUB_09') + prod_B_SUB(model,'SUB_09') <= 0.05)
model.prod_B_lb_SUB_16 = Constraint(expr=model.Mixer_Level_B * mixer_SUB(model, 'SUB_16') + prod_B_SUB(model,'SUB_16') >= 0.0055)
model.prod_B_ub_SUB_16 = Constraint(expr=model.Mixer_Level_B * mixer_SUB(model, 'SUB_16') + prod_B_SUB(model,'SUB_16') <= 0.018)
model.prod_B_ub_SUB_15 = Constraint(expr=model.Mixer_Level_B * mixer_SUB(model, 'SUB_15') + prod_B_SUB(model,'SUB_15') <= 0.7)
model.prod_B_lb_SUB_23 = Constraint(expr=model.Mixer_Level_B * mixer_SUB(model, 'SUB_23') + prod_B_SUB(model,'SUB_23') >= 0)
model.prod_B_ub_SUB_23 = Constraint(expr=model.Mixer_Level_B * mixer_SUB(model, 'SUB_23') + prod_B_SUB(model,'SUB_23') <= 0.16)

### MODEL EXPRESSIONS PROD C ###

# Define model_1C as an expression (SRI Model)
def model_1C(model):
    Total_Prod_C_SUB_01 = model.Mixer_Level_C * mixer_SUB(model, 'SUB_01') + prod_C_SUB(model,'SUB_01')
    Total_Prod_C_SUB_02 = model.Mixer_Level_C * mixer_SUB(model, 'SUB_02') + prod_C_SUB(model,'SUB_02')
    Total_Prod_C_SUB_09 = model.Mixer_Level_C * mixer_SUB(model, 'SUB_09') + prod_C_SUB(model,'SUB_09')
    Total_Prod_C_SUB_11 = model.Mixer_Level_C * mixer_SUB(model, 'SUB_11') + prod_C_SUB(model,'SUB_11')
    Total_Prod_C_SUB_18 = model.Mixer_Level_C * mixer_SUB(model, 'SUB_18') + prod_C_SUB(model,'SUB_18')
    Total_Prod_C_SUB_19 = model.Mixer_Level_C * mixer_SUB(model, 'SUB_19') + prod_C_SUB(model,'SUB_19')
    Total_Prod_C_SUB_27 = model.Mixer_Level_C * mixer_SUB(model, 'SUB_27') + prod_C_SUB(model,'SUB_27')

    return (
        53 - 6 * Total_Prod_C_SUB_11 - 5 * Total_Prod_C_SUB_02 + 1172 * Total_Prod_C_SUB_01 - 400 * Total_Prod_C_SUB_09 + 1874 * Total_Prod_C_SUB_18 +
        23 * Total_Prod_C_SUB_27 + 183510 * Total_Prod_C_SUB_19 - 2038 * Total_Prod_C_SUB_27 * Total_Prod_C_SUB_27 - 199367 * Total_Prod_C_SUB_18 * Total_Prod_C_SUB_18 +
        2561 * Total_Prod_C_SUB_27 * Total_Prod_C_SUB_09 + 130 * Total_Prod_C_SUB_11 * Total_Prod_C_SUB_09 + 2592 * Total_Prod_C_SUB_09 * Total_Prod_C_SUB_09 -
        103429177 * Total_Prod_C_SUB_19 * Total_Prod_C_SUB_19 - 1048878 * Total_Prod_C_SUB_19 * Total_Prod_C_SUB_18
    )

model.model_1C = Expression(rule=model_1C)

# Define model_2C as an expression (SRI Model)
def model_2C(model):
    Total_Prod_C_SUB_02 = model.Mixer_Level_C * mixer_SUB(model, 'SUB_02') + prod_C_SUB(model,'SUB_02')
    Total_Prod_C_SUB_09 = model.Mixer_Level_C * mixer_SUB(model, 'SUB_09') + prod_C_SUB(model,'SUB_09')
    Total_Prod_C_SUB_11 = model.Mixer_Level_C * mixer_SUB(model, 'SUB_11') + prod_C_SUB(model,'SUB_11')
    Total_Prod_C_SUB_15 = model.Mixer_Level_C * mixer_SUB(model, 'SUB_15') + prod_C_SUB(model,'SUB_15')
    Total_Prod_C_SUB_16 = model.Mixer_Level_C * mixer_SUB(model, 'SUB_16') + prod_C_SUB(model,'SUB_16')
    Total_Prod_C_SUB_21 = model.Mixer_Level_C * mixer_SUB(model, 'SUB_21') + prod_C_SUB(model,'SUB_21')
    Total_Prod_C_SUB_23 = model.Mixer_Level_C * mixer_SUB(model, 'SUB_23') + prod_C_SUB(model,'SUB_23')
    Total_Prod_C_SUB_24 = model.Mixer_Level_C * mixer_SUB(model, 'SUB_24') + prod_C_SUB(model,'SUB_24')
    Total_Prod_C_SUB_26 = model.Mixer_Level_C * mixer_SUB(model, 'SUB_26') + prod_C_SUB(model,'SUB_26')
    Total_Prod_C_SUB_27 = model.Mixer_Level_C * mixer_SUB(model, 'SUB_27') + prod_C_SUB(model,'SUB_27')

    return (
        29 + 51 * Total_Prod_C_SUB_21 + 42 * Total_Prod_C_SUB_15 + 15 * Total_Prod_C_SUB_23 + 25 * Total_Prod_C_SUB_11 + 51 * Total_Prod_C_SUB_02 + 121 * Total_Prod_C_SUB_16 -
        139 * Total_Prod_C_SUB_24 - 30 * Total_Prod_C_SUB_26 - 80 * Total_Prod_C_SUB_09 + 231 * Total_Prod_C_SUB_27 - 23 * Total_Prod_C_SUB_21 * Total_Prod_C_SUB_21 -
        36 * Total_Prod_C_SUB_21 * Total_Prod_C_SUB_15 - 27 * Total_Prod_C_SUB_21 * Total_Prod_C_SUB_23 - 21 * Total_Prod_C_SUB_21 * Total_Prod_C_SUB_02 -
        37 * Total_Prod_C_SUB_15 * Total_Prod_C_SUB_02 - 36 * Total_Prod_C_SUB_23 * Total_Prod_C_SUB_23 - 14 * Total_Prod_C_SUB_23 * Total_Prod_C_SUB_11 -
        1602 * Total_Prod_C_SUB_16 * Total_Prod_C_SUB_16 + 29 * Total_Prod_C_SUB_21 * Total_Prod_C_SUB_24 - 5 * Total_Prod_C_SUB_21 * Total_Prod_C_SUB_11 +
        178 * Total_Prod_C_SUB_24 * Total_Prod_C_SUB_16 + 54 * Total_Prod_C_SUB_24 * Total_Prod_C_SUB_26 - 25 * Total_Prod_C_SUB_15 * Total_Prod_C_SUB_11 -
        2912 * Total_Prod_C_SUB_27 * Total_Prod_C_SUB_27 + 68 * Total_Prod_C_SUB_27 * Total_Prod_C_SUB_11 - 20 * Total_Prod_C_SUB_11 * Total_Prod_C_SUB_02 +
        85 * Total_Prod_C_SUB_24 * Total_Prod_C_SUB_27 - 16 * Total_Prod_C_SUB_15 * Total_Prod_C_SUB_15 + 86 * Total_Prod_C_SUB_27 * Total_Prod_C_SUB_23 -
        50 * Total_Prod_C_SUB_27 * Total_Prod_C_SUB_02 + 222 * Total_Prod_C_SUB_27 * Total_Prod_C_SUB_26 + 1447 * Total_Prod_C_SUB_27 * Total_Prod_C_SUB_09 -
        57 * Total_Prod_C_SUB_11 * Total_Prod_C_SUB_16 + 37 * Total_Prod_C_SUB_11 * Total_Prod_C_SUB_26
    )

model.model_2C = Expression(rule=model_2C)

# Define model_3C as an expression (SRI Model)
def model_3C(model):
    Total_Prod_C_SUB_01 = model.Mixer_Level_C * mixer_SUB(model, 'SUB_01') + prod_C_SUB(model,'SUB_01')
    Total_Prod_C_SUB_02 = model.Mixer_Level_C * mixer_SUB(model, 'SUB_02') + prod_C_SUB(model,'SUB_02')
    Total_Prod_C_SUB_03 = model.Mixer_Level_C * mixer_SUB(model, 'SUB_03') + prod_C_SUB(model,'SUB_03')
    Total_Prod_C_SUB_11 = model.Mixer_Level_C * mixer_SUB(model, 'SUB_11') + prod_C_SUB(model,'SUB_11')
    Total_Prod_C_SUB_15 = model.Mixer_Level_C * mixer_SUB(model, 'SUB_15') + prod_C_SUB(model,'SUB_15')
    Total_Prod_C_SUB_18 = model.Mixer_Level_C * mixer_SUB(model, 'SUB_18') + prod_C_SUB(model,'SUB_18')
    Total_Prod_C_SUB_21 = model.Mixer_Level_C * mixer_SUB(model, 'SUB_21') + prod_C_SUB(model,'SUB_21')
    Total_Prod_C_SUB_23 = model.Mixer_Level_C * mixer_SUB(model, 'SUB_23') + prod_C_SUB(model,'SUB_23')
    Total_Prod_C_SUB_26 = model.Mixer_Level_C * mixer_SUB(model, 'SUB_26') + prod_C_SUB(model,'SUB_26')

    return (
        9 + 12 * Total_Prod_C_SUB_21 - 0.5 * Total_Prod_C_SUB_15 + 10 * Total_Prod_C_SUB_23 + 7 * Total_Prod_C_SUB_11 + 12 * Total_Prod_C_SUB_02 + 34 * Total_Prod_C_SUB_03 -
        13976 * Total_Prod_C_SUB_01 + 9 * Total_Prod_C_SUB_26 + 4069 * Total_Prod_C_SUB_18 - 8 * Total_Prod_C_SUB_21 * Total_Prod_C_SUB_21 - 2 * Total_Prod_C_SUB_21 * Total_Prod_C_SUB_15 -
        23 * Total_Prod_C_SUB_21 * Total_Prod_C_SUB_23 - 8 * Total_Prod_C_SUB_21 * Total_Prod_C_SUB_02 - 7 * Total_Prod_C_SUB_15 * Total_Prod_C_SUB_02 + 1754 * Total_Prod_C_SUB_15 * Total_Prod_C_SUB_18 +
        0.4 * Total_Prod_C_SUB_23 * Total_Prod_C_SUB_03 + 120 * Total_Prod_C_SUB_26 * Total_Prod_C_SUB_26 + 17776885 * Total_Prod_C_SUB_01 * Total_Prod_C_SUB_01 - 7 * Total_Prod_C_SUB_11 * Total_Prod_C_SUB_02 -
        364999 * Total_Prod_C_SUB_18 * Total_Prod_C_SUB_18 - 564 * Total_Prod_C_SUB_03 * Total_Prod_C_SUB_03
    )

model.model_3C = Expression(rule=model_3C)

# Define model_4C as a constraint based on model_1C, model_2C, and model_3C
def model_4C(model):
    return 0.3 * model.model_1C + 0.2 * model.model_2C + 0.5 * model.model_3C

# Define model_7C as an expression (Post Mixer Cost)
def model_7C(model):
    return sum(model.postmixer_costs[rm] * model.prod_C[rm] for rm in model.RM)

### MODEL CONSTRAINTS PROD C ###

# Total PROD C Composition Constraints
def prod_C_total_quantity_rule(model):
    return sum(model.prod_C[rm] for rm in model.RM) + model.Mixer_Level_C == 1.0
model.prod_C_total_quantity_constraint = Constraint(rule=prod_C_total_quantity_rule)

# model_4C constraint for the SRI
model.model_4C_constraint = Constraint(expr=model_4C(model) >= 44)

# Component Constraint in the Post Mixer
model.prod_C_RM_01 = Constraint(expr=model.prod_C['RM_01'] == 0)
model.prod_C_RM_02 = Constraint(expr=model.prod_C['RM_02'] == 0)
model.prod_C_RM_05 = Constraint(expr=model.prod_C['RM_05'] == 0)
model.prod_C_RM_06 = Constraint(expr=model.prod_C['RM_06'] == 0)
model.prod_C_RM_10 = Constraint(expr=model.prod_C['RM_10'] == 0)
model.prod_C_RM_14 = Constraint(expr=model.prod_C['RM_14'] == 0)
model.prod_C_RM_15 = Constraint(expr=model.prod_C['RM_15'] == 0)
model.prod_C_RM_16 = Constraint(expr=model.prod_C['RM_16'] == 0)
model.prod_C_RM_18 = Constraint(expr=model.prod_C['RM_18'] == 0)
model.prod_C_RM_19 = Constraint(expr=model.prod_C['RM_19'] == 0)
model.prod_C_RM_21 = Constraint(expr=model.prod_C['RM_21'] == 0)
model.prod_C_RM_22 = Constraint(expr=model.prod_C['RM_22'] == 0)

# Final Products RM Component Bounds
model.prod_C_lb_RM_12 = Constraint(expr=model.Mixer_Level_C *model.Mixer_RM_qty['RM_12'] + model.prod_C['RM_12'] >= 0.008)
model.prod_C_ub_RM_18 = Constraint(expr=model.Mixer_Level_C *model.Mixer_RM_qty['RM_18'] + model.prod_C['RM_18'] <= 0.02)
model.prod_C_ub_RM_08 = Constraint(expr=model.Mixer_Level_C *model.Mixer_RM_qty['RM_08'] + model.prod_C['RM_08'] <= 0.46)

# Final Products SUB Component Bounds
model.prod_C_lb_SUB_21 = Constraint(expr=model.Mixer_Level_C * mixer_SUB(model, 'SUB_21') + prod_C_SUB(model,'SUB_21') >= 0)
model.prod_C_ub_SUB_21 = Constraint(expr=model.Mixer_Level_C * mixer_SUB(model, 'SUB_21') + prod_C_SUB(model,'SUB_21') <= 0.75)
model.prod_C_lb_SUB_27 = Constraint(expr=model.Mixer_Level_C * mixer_SUB(model, 'SUB_27') + prod_C_SUB(model,'SUB_27') >= 0.003)
model.prod_C_ub_SUB_27 = Constraint(expr=model.Mixer_Level_C * mixer_SUB(model, 'SUB_27') + prod_C_SUB(model,'SUB_27') <= 0.02)
model.prod_C_ub_SUB_03 = Constraint(expr=model.Mixer_Level_C * mixer_SUB(model, 'SUB_03') + prod_C_SUB(model,'SUB_03') <= 0.09)
model.prod_C_lb_SUB_26 = Constraint(expr=model.Mixer_Level_C * mixer_SUB(model, 'SUB_26') + prod_C_SUB(model,'SUB_26') >= 0.02)
model.prod_C_ub_SUB_26 = Constraint(expr=model.Mixer_Level_C * mixer_SUB(model, 'SUB_26') + prod_C_SUB(model,'SUB_26') <= 0.2)
model.prod_C_lb_SUB_20 = Constraint(expr=model.Mixer_Level_C * mixer_SUB(model, 'SUB_20') + prod_C_SUB(model,'SUB_20') >= 0.001)
model.prod_C_lb_SUB_19 = Constraint(expr=model.Mixer_Level_C * mixer_SUB(model, 'SUB_19') + prod_C_SUB(model,'SUB_19') >= 0.0001)
model.prod_C_ub_SUB_19 = Constraint(expr=model.Mixer_Level_C * mixer_SUB(model, 'SUB_19') + prod_C_SUB(model,'SUB_19') <= 0.0009)
model.prod_C_lb_SUB_18 = Constraint(expr=model.Mixer_Level_C * mixer_SUB(model, 'SUB_18') + prod_C_SUB(model,'SUB_18') >= 0.0001)
model.prod_C_ub_SUB_18 = Constraint(expr=model.Mixer_Level_C * mixer_SUB(model, 'SUB_18') + prod_C_SUB(model,'SUB_18') <= 0.0065)
model.prod_C_ub_SUB_10 = Constraint(expr=model.Mixer_Level_C * mixer_SUB(model, 'SUB_10') + prod_C_SUB(model,'SUB_10') <= 0.029)
model.prod_C_lb_SUB_24 = Constraint(expr=model.Mixer_Level_C * mixer_SUB(model, 'SUB_24') + prod_C_SUB(model,'SUB_24') >= 0)
model.prod_C_ub_SUB_24 = Constraint(expr=model.Mixer_Level_C * mixer_SUB(model, 'SUB_24') + prod_C_SUB(model,'SUB_24') <= 0.2)
model.prod_C_lb_SUB_09 = Constraint(expr=model.Mixer_Level_C * mixer_SUB(model, 'SUB_09') + prod_C_SUB(model,'SUB_09') >= 0.002)
model.prod_C_ub_SUB_09 = Constraint(expr=model.Mixer_Level_C * mixer_SUB(model, 'SUB_09') + prod_C_SUB(model,'SUB_09') <= 0.05)
model.prod_C_lb_SUB_16 = Constraint(expr=model.Mixer_Level_C * mixer_SUB(model, 'SUB_16') + prod_C_SUB(model,'SUB_16') >= 0.0055)
model.prod_C_ub_SUB_16 = Constraint(expr=model.Mixer_Level_C * mixer_SUB(model, 'SUB_16') + prod_C_SUB(model,'SUB_16') <= 0.018)
model.prod_C_ub_SUB_15 = Constraint(expr=model.Mixer_Level_C * mixer_SUB(model, 'SUB_15') + prod_C_SUB(model,'SUB_15') <= 0.7)
model.prod_C_lb_SUB_23 = Constraint(expr=model.Mixer_Level_C * mixer_SUB(model, 'SUB_23') + prod_C_SUB(model,'SUB_23') >= 0)
model.prod_C_ub_SUB_23 = Constraint(expr=model.Mixer_Level_C * mixer_SUB(model, 'SUB_23') + prod_C_SUB(model,'SUB_23') <= 0.16)

### MODEL EXPRESSIONS PROD D ###

# Define model_1D as an expression (SRI Model)
def model_1D(model):
    Total_Prod_D_SUB_01 = model.Mixer_Level_D * mixer_SUB(model, 'SUB_01') + prod_D_SUB(model,'SUB_01')
    Total_Prod_D_SUB_02 = model.Mixer_Level_D * mixer_SUB(model, 'SUB_02') + prod_D_SUB(model,'SUB_02')
    Total_Prod_D_SUB_09 = model.Mixer_Level_D * mixer_SUB(model, 'SUB_09') + prod_D_SUB(model,'SUB_09')
    Total_Prod_D_SUB_11 = model.Mixer_Level_D * mixer_SUB(model, 'SUB_11') + prod_D_SUB(model,'SUB_11')
    Total_Prod_D_SUB_18 = model.Mixer_Level_D * mixer_SUB(model, 'SUB_18') + prod_D_SUB(model,'SUB_18')
    Total_Prod_D_SUB_19 = model.Mixer_Level_D * mixer_SUB(model, 'SUB_19') + prod_D_SUB(model,'SUB_19')
    Total_Prod_D_SUB_27 = model.Mixer_Level_D * mixer_SUB(model, 'SUB_27') + prod_D_SUB(model,'SUB_27')

    return (
        53 - 6 * Total_Prod_D_SUB_11 - 5 * Total_Prod_D_SUB_02 + 1172 * Total_Prod_D_SUB_01 - 400 * Total_Prod_D_SUB_09 + 1874 * Total_Prod_D_SUB_18 +
        23 * Total_Prod_D_SUB_27 + 183510 * Total_Prod_D_SUB_19 - 2038 * Total_Prod_D_SUB_27 * Total_Prod_D_SUB_27 - 199367 * Total_Prod_D_SUB_18 * Total_Prod_D_SUB_18 +
        2561 * Total_Prod_D_SUB_27 * Total_Prod_D_SUB_09 + 130 * Total_Prod_D_SUB_11 * Total_Prod_D_SUB_09 + 2592 * Total_Prod_D_SUB_09 * Total_Prod_D_SUB_09 -
        103429177 * Total_Prod_D_SUB_19 * Total_Prod_D_SUB_19 - 1048878 * Total_Prod_D_SUB_19 * Total_Prod_D_SUB_18
    )

model.model_1D = Expression(rule=model_1D)

# Define model_2D as an expression (SRI Model)
def model_2D(model):
    Total_Prod_D_SUB_02 = model.Mixer_Level_D * mixer_SUB(model, 'SUB_02') + prod_D_SUB(model,'SUB_02')
    Total_Prod_D_SUB_09 = model.Mixer_Level_D * mixer_SUB(model, 'SUB_09') + prod_D_SUB(model,'SUB_09')
    Total_Prod_D_SUB_11 = model.Mixer_Level_D * mixer_SUB(model, 'SUB_11') + prod_D_SUB(model,'SUB_11')
    Total_Prod_D_SUB_15 = model.Mixer_Level_D * mixer_SUB(model, 'SUB_15') + prod_D_SUB(model,'SUB_15')
    Total_Prod_D_SUB_16 = model.Mixer_Level_D * mixer_SUB(model, 'SUB_16') + prod_D_SUB(model,'SUB_16')
    Total_Prod_D_SUB_21 = model.Mixer_Level_D * mixer_SUB(model, 'SUB_21') + prod_D_SUB(model,'SUB_21')
    Total_Prod_D_SUB_23 = model.Mixer_Level_D * mixer_SUB(model, 'SUB_23') + prod_D_SUB(model,'SUB_23')
    Total_Prod_D_SUB_24 = model.Mixer_Level_D * mixer_SUB(model, 'SUB_24') + prod_D_SUB(model,'SUB_24')
    Total_Prod_D_SUB_26 = model.Mixer_Level_D * mixer_SUB(model, 'SUB_26') + prod_D_SUB(model,'SUB_26')
    Total_Prod_D_SUB_27 = model.Mixer_Level_D * mixer_SUB(model, 'SUB_27') + prod_D_SUB(model,'SUB_27')

    return (
        29 + 51 * Total_Prod_D_SUB_21 + 42 * Total_Prod_D_SUB_15 + 15 * Total_Prod_D_SUB_23 + 25 * Total_Prod_D_SUB_11 + 51 * Total_Prod_D_SUB_02 + 121 * Total_Prod_D_SUB_16 -
        139 * Total_Prod_D_SUB_24 - 30 * Total_Prod_D_SUB_26 - 80 * Total_Prod_D_SUB_09 + 231 * Total_Prod_D_SUB_27 - 23 * Total_Prod_D_SUB_21 * Total_Prod_D_SUB_21 -
        36 * Total_Prod_D_SUB_21 * Total_Prod_D_SUB_15 - 27 * Total_Prod_D_SUB_21 * Total_Prod_D_SUB_23 - 21 * Total_Prod_D_SUB_21 * Total_Prod_D_SUB_02 -
        37 * Total_Prod_D_SUB_15 * Total_Prod_D_SUB_02 - 36 * Total_Prod_D_SUB_23 * Total_Prod_D_SUB_23 - 14 * Total_Prod_D_SUB_23 * Total_Prod_D_SUB_11 -
        1602 * Total_Prod_D_SUB_16 * Total_Prod_D_SUB_16 + 29 * Total_Prod_D_SUB_21 * Total_Prod_D_SUB_24 - 5 * Total_Prod_D_SUB_21 * Total_Prod_D_SUB_11 +
        178 * Total_Prod_D_SUB_24 * Total_Prod_D_SUB_16 + 54 * Total_Prod_D_SUB_24 * Total_Prod_D_SUB_26 - 25 * Total_Prod_D_SUB_15 * Total_Prod_D_SUB_11 -
        2912 * Total_Prod_D_SUB_27 * Total_Prod_D_SUB_27 + 68 * Total_Prod_D_SUB_27 * Total_Prod_D_SUB_11 - 20 * Total_Prod_D_SUB_11 * Total_Prod_D_SUB_02 +
        85 * Total_Prod_D_SUB_24 * Total_Prod_D_SUB_27 - 16 * Total_Prod_D_SUB_15 * Total_Prod_D_SUB_15 + 86 * Total_Prod_D_SUB_27 * Total_Prod_D_SUB_23 -
        50 * Total_Prod_D_SUB_27 * Total_Prod_D_SUB_02 + 222 * Total_Prod_D_SUB_27 * Total_Prod_D_SUB_26 + 1447 * Total_Prod_D_SUB_27 * Total_Prod_D_SUB_09 -
        57 * Total_Prod_D_SUB_11 * Total_Prod_D_SUB_16 + 37 * Total_Prod_D_SUB_11 * Total_Prod_D_SUB_26
    )

model.model_2D = Expression(rule=model_2D)

# Define model_3D as an expression (SRI Model)
def model_3D(model):
    Total_Prod_D_SUB_01 = model.Mixer_Level_D * mixer_SUB(model, 'SUB_01') + prod_D_SUB(model,'SUB_01')
    Total_Prod_D_SUB_02 = model.Mixer_Level_D * mixer_SUB(model, 'SUB_02') + prod_D_SUB(model,'SUB_02')
    Total_Prod_D_SUB_03 = model.Mixer_Level_D * mixer_SUB(model, 'SUB_03') + prod_D_SUB(model,'SUB_03')
    Total_Prod_D_SUB_11 = model.Mixer_Level_D * mixer_SUB(model, 'SUB_11') + prod_D_SUB(model,'SUB_11')
    Total_Prod_D_SUB_15 = model.Mixer_Level_D * mixer_SUB(model, 'SUB_15') + prod_D_SUB(model,'SUB_15')
    Total_Prod_D_SUB_18 = model.Mixer_Level_D * mixer_SUB(model, 'SUB_18') + prod_D_SUB(model,'SUB_18')
    Total_Prod_D_SUB_21 = model.Mixer_Level_D * mixer_SUB(model, 'SUB_21') + prod_D_SUB(model,'SUB_21')
    Total_Prod_D_SUB_23 = model.Mixer_Level_D * mixer_SUB(model, 'SUB_23') + prod_D_SUB(model,'SUB_23')
    Total_Prod_D_SUB_26 = model.Mixer_Level_D * mixer_SUB(model, 'SUB_26') + prod_D_SUB(model,'SUB_26')

    return (
        9 + 12 * Total_Prod_D_SUB_21 - 0.5 * Total_Prod_D_SUB_15 + 10 * Total_Prod_D_SUB_23 + 7 * Total_Prod_D_SUB_11 + 12 * Total_Prod_D_SUB_02 + 34 * Total_Prod_D_SUB_03 -
        13976 * Total_Prod_D_SUB_01 + 9 * Total_Prod_D_SUB_26 + 4069 * Total_Prod_D_SUB_18 - 8 * Total_Prod_D_SUB_21 * Total_Prod_D_SUB_21 - 2 * Total_Prod_D_SUB_21 * Total_Prod_D_SUB_15 -
        23 * Total_Prod_D_SUB_21 * Total_Prod_D_SUB_23 - 8 * Total_Prod_D_SUB_21 * Total_Prod_D_SUB_02 - 7 * Total_Prod_D_SUB_15 * Total_Prod_D_SUB_02 + 1754 * Total_Prod_D_SUB_15 * Total_Prod_D_SUB_18 +
        0.4 * Total_Prod_D_SUB_23 * Total_Prod_D_SUB_03 + 120 * Total_Prod_D_SUB_26 * Total_Prod_D_SUB_26 + 17776885 * Total_Prod_D_SUB_01 * Total_Prod_D_SUB_01 - 7 * Total_Prod_D_SUB_11 * Total_Prod_D_SUB_02 -
        364999 * Total_Prod_D_SUB_18 * Total_Prod_D_SUB_18 - 564 * Total_Prod_D_SUB_03 * Total_Prod_D_SUB_03
    )

model.model_3D = Expression(rule=model_3D)

# Define model_4D as a Constraint based on model_1D, model_2D, and model_3D
def model_4D(model):
    return 0.3 * model.model_1D + 0.2 * model.model_2D + 0.5 * model.model_3D

# Define model_7D as an expression (Post Mixer cost)
def model_7D(model):
    return sum(model.postmixer_costs[rm] * model.prod_D[rm] for rm in model.RM)

### MODEL ConstraintS PROD D ###

# Total PROD D Domposition Constraints
def prod_D_total_quantity_rule(model):
    return sum(model.prod_D[rm] for rm in model.RM) + model.Mixer_Level_D == 1.0
model.prod_D_total_quantity_Constraint = Constraint(rule=prod_D_total_quantity_rule)

# model_4D Constraint for the SRI
model.model_4D_Constraint = Constraint(expr=model_4D(model) >= 53)

# Component Constraint in the Post Mixer
model.prod_D_RM_01 = Constraint(expr=model.prod_D['RM_01'] == 0)
model.prod_D_RM_02 = Constraint(expr=model.prod_D['RM_02'] == 0)
model.prod_D_RM_05 = Constraint(expr=model.prod_D['RM_05'] == 0)
model.prod_D_RM_06 = Constraint(expr=model.prod_D['RM_06'] == 0)
model.prod_D_RM_10 = Constraint(expr=model.prod_D['RM_10'] == 0)
model.prod_D_RM_14 = Constraint(expr=model.prod_D['RM_14'] == 0)
model.prod_D_RM_15 = Constraint(expr=model.prod_D['RM_15'] == 0)
model.prod_D_RM_16 = Constraint(expr=model.prod_D['RM_16'] == 0)
model.prod_D_RM_18 = Constraint(expr=model.prod_D['RM_18'] == 0)
model.prod_D_RM_19 = Constraint(expr=model.prod_D['RM_19'] == 0)
model.prod_D_RM_21 = Constraint(expr=model.prod_D['RM_21'] == 0)
model.prod_D_RM_22 = Constraint(expr=model.prod_D['RM_22'] == 0)

# Final Products RM Component Bounds
model.prod_D_lb_RM_12 = Constraint(expr=model.Mixer_Level_D *model.Mixer_RM_qty['RM_12'] + model.prod_D['RM_12'] >= 0.008)
model.prod_D_ub_RM_18 = Constraint(expr=model.Mixer_Level_D *model.Mixer_RM_qty['RM_18'] + model.prod_D['RM_18'] <= 0.02)
model.prod_D_ub_RM_08 = Constraint(expr=model.Mixer_Level_D *model.Mixer_RM_qty['RM_08'] + model.prod_D['RM_08'] <= 0.46)

# Final Products SUB Component Bounds
model.prod_D_lb_SUB_21 = Constraint(expr=model.Mixer_Level_D * mixer_SUB(model, 'SUB_21') + prod_D_SUB(model,'SUB_21') >= 0)
model.prod_D_ub_SUB_21 = Constraint(expr=model.Mixer_Level_D * mixer_SUB(model, 'SUB_21') + prod_D_SUB(model,'SUB_21') <= 0.75)
model.prod_D_lb_SUB_27 = Constraint(expr=model.Mixer_Level_D * mixer_SUB(model, 'SUB_27') + prod_D_SUB(model,'SUB_27') >= 0.003)
model.prod_D_ub_SUB_27 = Constraint(expr=model.Mixer_Level_D * mixer_SUB(model, 'SUB_27') + prod_D_SUB(model,'SUB_27') <= 0.02)
model.prod_D_ub_SUB_03 = Constraint(expr=model.Mixer_Level_D * mixer_SUB(model, 'SUB_03') + prod_D_SUB(model,'SUB_03') <= 0.09)
model.prod_D_lb_SUB_26 = Constraint(expr=model.Mixer_Level_D * mixer_SUB(model, 'SUB_26') + prod_D_SUB(model,'SUB_26') >= 0.02)
model.prod_D_ub_SUB_26 = Constraint(expr=model.Mixer_Level_D * mixer_SUB(model, 'SUB_26') + prod_D_SUB(model,'SUB_26') <= 0.2)
model.prod_D_lb_SUB_20 = Constraint(expr=model.Mixer_Level_D * mixer_SUB(model, 'SUB_20') + prod_D_SUB(model,'SUB_20') >= 0.001)
model.prod_D_lb_SUB_19 = Constraint(expr=model.Mixer_Level_D * mixer_SUB(model, 'SUB_19') + prod_D_SUB(model,'SUB_19') >= 0.0001)
model.prod_D_ub_SUB_19 = Constraint(expr=model.Mixer_Level_D * mixer_SUB(model, 'SUB_19') + prod_D_SUB(model,'SUB_19') <= 0.0009)
model.prod_D_lb_SUB_18 = Constraint(expr=model.Mixer_Level_D * mixer_SUB(model, 'SUB_18') + prod_D_SUB(model,'SUB_18') >= 0.0001)
model.prod_D_ub_SUB_18 = Constraint(expr=model.Mixer_Level_D * mixer_SUB(model, 'SUB_18') + prod_D_SUB(model,'SUB_18') <= 0.0065)
model.prod_D_ub_SUB_10 = Constraint(expr=model.Mixer_Level_D * mixer_SUB(model, 'SUB_10') + prod_D_SUB(model,'SUB_10') <= 0.029)
model.prod_D_lb_SUB_24 = Constraint(expr=model.Mixer_Level_D * mixer_SUB(model, 'SUB_24') + prod_D_SUB(model,'SUB_24') >= 0)
model.prod_D_ub_SUB_24 = Constraint(expr=model.Mixer_Level_D * mixer_SUB(model, 'SUB_24') + prod_D_SUB(model,'SUB_24') <= 0.2)
model.prod_D_lb_SUB_09 = Constraint(expr=model.Mixer_Level_D * mixer_SUB(model, 'SUB_09') + prod_D_SUB(model,'SUB_09') >= 0.002)
model.prod_D_ub_SUB_09 = Constraint(expr=model.Mixer_Level_D * mixer_SUB(model, 'SUB_09') + prod_D_SUB(model,'SUB_09') <= 0.05)
model.prod_D_lb_SUB_16 = Constraint(expr=model.Mixer_Level_D * mixer_SUB(model, 'SUB_16') + prod_D_SUB(model,'SUB_16') >= 0.0055)
model.prod_D_ub_SUB_16 = Constraint(expr=model.Mixer_Level_D * mixer_SUB(model, 'SUB_16') + prod_D_SUB(model,'SUB_16') <= 0.018)
model.prod_D_ub_SUB_15 = Constraint(expr=model.Mixer_Level_D * mixer_SUB(model, 'SUB_15') + prod_D_SUB(model,'SUB_15') <= 0.7)
model.prod_D_lb_SUB_23 = Constraint(expr=model.Mixer_Level_D * mixer_SUB(model, 'SUB_23') + prod_D_SUB(model,'SUB_23') >= 0)
model.prod_D_ub_SUB_23 = Constraint(expr=model.Mixer_Level_D * mixer_SUB(model, 'SUB_23') + prod_D_SUB(model,'SUB_23') <= 0.16)

### MODEL EXPRESSIONS PROD E ###

# Define model_1E as an expression (SRI Model)
def model_1E(model):
    Total_Prod_E_SUB_01 = model.Mixer_Level_E * mixer_SUB(model, 'SUB_01') + prod_E_SUB(model,'SUB_01')
    Total_Prod_E_SUB_02 = model.Mixer_Level_E * mixer_SUB(model, 'SUB_02') + prod_E_SUB(model,'SUB_02')
    Total_Prod_E_SUB_09 = model.Mixer_Level_E * mixer_SUB(model, 'SUB_09') + prod_E_SUB(model,'SUB_09')
    Total_Prod_E_SUB_11 = model.Mixer_Level_E * mixer_SUB(model, 'SUB_11') + prod_E_SUB(model,'SUB_11')
    Total_Prod_E_SUB_18 = model.Mixer_Level_E * mixer_SUB(model, 'SUB_18') + prod_E_SUB(model,'SUB_18')
    Total_Prod_E_SUB_19 = model.Mixer_Level_E * mixer_SUB(model, 'SUB_19') + prod_E_SUB(model,'SUB_19')
    Total_Prod_E_SUB_27 = model.Mixer_Level_E * mixer_SUB(model, 'SUB_27') + prod_E_SUB(model,'SUB_27')

    return (
        53 - 6 * Total_Prod_E_SUB_11 - 5 * Total_Prod_E_SUB_02 + 1172 * Total_Prod_E_SUB_01 - 400 * Total_Prod_E_SUB_09 + 1874 * Total_Prod_E_SUB_18 +
        23 * Total_Prod_E_SUB_27 + 183510 * Total_Prod_E_SUB_19 - 2038 * Total_Prod_E_SUB_27 * Total_Prod_E_SUB_27 - 199367 * Total_Prod_E_SUB_18 * Total_Prod_E_SUB_18 +
        2561 * Total_Prod_E_SUB_27 * Total_Prod_E_SUB_09 + 130 * Total_Prod_E_SUB_11 * Total_Prod_E_SUB_09 + 2592 * Total_Prod_E_SUB_09 * Total_Prod_E_SUB_09 -
        103429177 * Total_Prod_E_SUB_19 * Total_Prod_E_SUB_19 - 1048878 * Total_Prod_E_SUB_19 * Total_Prod_E_SUB_18
    )

model.model_1E = Expression(rule=model_1E)

# Define model_2E as an expression (SRI Model)
def model_2E(model):
    Total_Prod_E_SUB_02 = model.Mixer_Level_E * mixer_SUB(model, 'SUB_02') + prod_E_SUB(model,'SUB_02')
    Total_Prod_E_SUB_09 = model.Mixer_Level_E * mixer_SUB(model, 'SUB_09') + prod_E_SUB(model,'SUB_09')
    Total_Prod_E_SUB_11 = model.Mixer_Level_E * mixer_SUB(model, 'SUB_11') + prod_E_SUB(model,'SUB_11')
    Total_Prod_E_SUB_15 = model.Mixer_Level_E * mixer_SUB(model, 'SUB_15') + prod_E_SUB(model,'SUB_15')
    Total_Prod_E_SUB_16 = model.Mixer_Level_E * mixer_SUB(model, 'SUB_16') + prod_E_SUB(model,'SUB_16')
    Total_Prod_E_SUB_21 = model.Mixer_Level_E * mixer_SUB(model, 'SUB_21') + prod_E_SUB(model,'SUB_21')
    Total_Prod_E_SUB_23 = model.Mixer_Level_E * mixer_SUB(model, 'SUB_23') + prod_E_SUB(model,'SUB_23')
    Total_Prod_E_SUB_24 = model.Mixer_Level_E * mixer_SUB(model, 'SUB_24') + prod_E_SUB(model,'SUB_24')
    Total_Prod_E_SUB_26 = model.Mixer_Level_E * mixer_SUB(model, 'SUB_26') + prod_E_SUB(model,'SUB_26')
    Total_Prod_E_SUB_27 = model.Mixer_Level_E * mixer_SUB(model, 'SUB_27') + prod_E_SUB(model,'SUB_27')

    return (
        29 + 51 * Total_Prod_E_SUB_21 + 42 * Total_Prod_E_SUB_15 + 15 * Total_Prod_E_SUB_23 + 25 * Total_Prod_E_SUB_11 + 51 * Total_Prod_E_SUB_02 + 121 * Total_Prod_E_SUB_16 -
        139 * Total_Prod_E_SUB_24 - 30 * Total_Prod_E_SUB_26 - 80 * Total_Prod_E_SUB_09 + 231 * Total_Prod_E_SUB_27 - 23 * Total_Prod_E_SUB_21 * Total_Prod_E_SUB_21 -
        36 * Total_Prod_E_SUB_21 * Total_Prod_E_SUB_15 - 27 * Total_Prod_E_SUB_21 * Total_Prod_E_SUB_23 - 21 * Total_Prod_E_SUB_21 * Total_Prod_E_SUB_02 -
        37 * Total_Prod_E_SUB_15 * Total_Prod_E_SUB_02 - 36 * Total_Prod_E_SUB_23 * Total_Prod_E_SUB_23 - 14 * Total_Prod_E_SUB_23 * Total_Prod_E_SUB_11 -
        1602 * Total_Prod_E_SUB_16 * Total_Prod_E_SUB_16 + 29 * Total_Prod_E_SUB_21 * Total_Prod_E_SUB_24 - 5 * Total_Prod_E_SUB_21 * Total_Prod_E_SUB_11 +
        178 * Total_Prod_E_SUB_24 * Total_Prod_E_SUB_16 + 54 * Total_Prod_E_SUB_24 * Total_Prod_E_SUB_26 - 25 * Total_Prod_E_SUB_15 * Total_Prod_E_SUB_11 -
        2912 * Total_Prod_E_SUB_27 * Total_Prod_E_SUB_27 + 68 * Total_Prod_E_SUB_27 * Total_Prod_E_SUB_11 - 20 * Total_Prod_E_SUB_11 * Total_Prod_E_SUB_02 +
        85 * Total_Prod_E_SUB_24 * Total_Prod_E_SUB_27 - 16 * Total_Prod_E_SUB_15 * Total_Prod_E_SUB_15 + 86 * Total_Prod_E_SUB_27 * Total_Prod_E_SUB_23 -
        50 * Total_Prod_E_SUB_27 * Total_Prod_E_SUB_02 + 222 * Total_Prod_E_SUB_27 * Total_Prod_E_SUB_26 + 1447 * Total_Prod_E_SUB_27 * Total_Prod_E_SUB_09 -
        57 * Total_Prod_E_SUB_11 * Total_Prod_E_SUB_16 + 37 * Total_Prod_E_SUB_11 * Total_Prod_E_SUB_26
    )

model.model_2E = Expression(rule=model_2E)

# Define model_3E as an expression (SRI Model)
def model_3E(model):
    Total_Prod_E_SUB_01 = model.Mixer_Level_E * mixer_SUB(model, 'SUB_01') + prod_E_SUB(model,'SUB_01')
    Total_Prod_E_SUB_02 = model.Mixer_Level_E * mixer_SUB(model, 'SUB_02') + prod_E_SUB(model,'SUB_02')
    Total_Prod_E_SUB_03 = model.Mixer_Level_E * mixer_SUB(model, 'SUB_03') + prod_E_SUB(model,'SUB_03')
    Total_Prod_E_SUB_11 = model.Mixer_Level_E * mixer_SUB(model, 'SUB_11') + prod_E_SUB(model,'SUB_11')
    Total_Prod_E_SUB_15 = model.Mixer_Level_E * mixer_SUB(model, 'SUB_15') + prod_E_SUB(model,'SUB_15')
    Total_Prod_E_SUB_18 = model.Mixer_Level_E * mixer_SUB(model, 'SUB_18') + prod_E_SUB(model,'SUB_18')
    Total_Prod_E_SUB_21 = model.Mixer_Level_E * mixer_SUB(model, 'SUB_21') + prod_E_SUB(model,'SUB_21')
    Total_Prod_E_SUB_23 = model.Mixer_Level_E * mixer_SUB(model, 'SUB_23') + prod_E_SUB(model,'SUB_23')
    Total_Prod_E_SUB_26 = model.Mixer_Level_E * mixer_SUB(model, 'SUB_26') + prod_E_SUB(model,'SUB_26')

    return (
        9 + 12 * Total_Prod_E_SUB_21 - 0.5 * Total_Prod_E_SUB_15 + 10 * Total_Prod_E_SUB_23 + 7 * Total_Prod_E_SUB_11 + 12 * Total_Prod_E_SUB_02 + 34 * Total_Prod_E_SUB_03 -
        13976 * Total_Prod_E_SUB_01 + 9 * Total_Prod_E_SUB_26 + 4069 * Total_Prod_E_SUB_18 - 8 * Total_Prod_E_SUB_21 * Total_Prod_E_SUB_21 - 2 * Total_Prod_E_SUB_21 * Total_Prod_E_SUB_15 -
        23 * Total_Prod_E_SUB_21 * Total_Prod_E_SUB_23 - 8 * Total_Prod_E_SUB_21 * Total_Prod_E_SUB_02 - 7 * Total_Prod_E_SUB_15 * Total_Prod_E_SUB_02 + 1754 * Total_Prod_E_SUB_15 * Total_Prod_E_SUB_18 +
        0.4 * Total_Prod_E_SUB_23 * Total_Prod_E_SUB_03 + 120 * Total_Prod_E_SUB_26 * Total_Prod_E_SUB_26 + 17776885 * Total_Prod_E_SUB_01 * Total_Prod_E_SUB_01 - 7 * Total_Prod_E_SUB_11 * Total_Prod_E_SUB_02 -
        364999 * Total_Prod_E_SUB_18 * Total_Prod_E_SUB_18 - 564 * Total_Prod_E_SUB_03 * Total_Prod_E_SUB_03
    )

model.model_3E = Expression(rule=model_3E)

# Define model_4E as a Constraint based on model_1E, model_2E, and model_3E
def model_4E(model):
    return 0.3 * model.model_1E + 0.2 * model.model_2E + 0.5 * model.model_3E

# Define model_7E as an expression (Post Mixer Cost)
def model_7E(model):
    return sum(model.postmixer_costs[rm] * model.prod_E[rm] for rm in model.RM)

### MODEL ConstraintS PROD E ###

# Total PROD E Domposition Constraints
def prod_E_total_quantity_rule(model):
    return sum(model.prod_E[rm] for rm in model.RM) + model.Mixer_Level_E == 1.0
model.prod_E_total_quantity_Constraint = Constraint(rule=prod_E_total_quantity_rule)

# model_4E Constraint for the SRI
model.model_4E_Constraint = Constraint(expr=model_4D(model) >= 48)

# Component Constraint in the Post Mixer
model.prod_E_RM_01 = Constraint(expr=model.prod_E['RM_01'] == 0)
model.prod_E_RM_02 = Constraint(expr=model.prod_E['RM_02'] == 0)
model.prod_E_RM_05 = Constraint(expr=model.prod_E['RM_05'] == 0)
model.prod_E_RM_06 = Constraint(expr=model.prod_E['RM_06'] == 0)
model.prod_E_RM_10 = Constraint(expr=model.prod_E['RM_10'] == 0)
model.prod_E_RM_14 = Constraint(expr=model.prod_E['RM_14'] == 0)
model.prod_E_RM_15 = Constraint(expr=model.prod_E['RM_15'] == 0)
model.prod_E_RM_16 = Constraint(expr=model.prod_E['RM_16'] == 0)
model.prod_E_RM_18 = Constraint(expr=model.prod_E['RM_18'] == 0)
model.prod_E_RM_19 = Constraint(expr=model.prod_E['RM_19'] == 0)
model.prod_E_RM_21 = Constraint(expr=model.prod_E['RM_21'] == 0)
model.prod_E_RM_22 = Constraint(expr=model.prod_E['RM_22'] == 0)

# Final Products RM Component Bounds
model.prod_E_lb_RM_12 = Constraint(expr=model.Mixer_Level_E *model.Mixer_RM_qty['RM_12'] + model.prod_E['RM_12'] >= 0.008)
model.prod_E_ub_RM_18 = Constraint(expr=model.Mixer_Level_E *model.Mixer_RM_qty['RM_18'] + model.prod_E['RM_18'] <= 0.02)
model.prod_E_ub_RM_08 = Constraint(expr=model.Mixer_Level_E *model.Mixer_RM_qty['RM_08'] + model.prod_E['RM_08'] <= 0.46)

# Final Products SUB Component Bounds
model.prod_E_lb_SUB_21 = Constraint(expr=model.Mixer_Level_E * mixer_SUB(model, 'SUB_21') + prod_E_SUB(model,'SUB_21') >= 0)
model.prod_E_ub_SUB_21 = Constraint(expr=model.Mixer_Level_E * mixer_SUB(model, 'SUB_21') + prod_E_SUB(model,'SUB_21') <= 0.75)
model.prod_E_lb_SUB_27 = Constraint(expr=model.Mixer_Level_E * mixer_SUB(model, 'SUB_27') + prod_E_SUB(model,'SUB_27') >= 0.003)
model.prod_E_ub_SUB_27 = Constraint(expr=model.Mixer_Level_E * mixer_SUB(model, 'SUB_27') + prod_E_SUB(model,'SUB_27') <= 0.02)
model.prod_E_ub_SUB_03 = Constraint(expr=model.Mixer_Level_E * mixer_SUB(model, 'SUB_03') + prod_E_SUB(model,'SUB_03') <= 0.09)
model.prod_E_lb_SUB_26 = Constraint(expr=model.Mixer_Level_E * mixer_SUB(model, 'SUB_26') + prod_E_SUB(model,'SUB_26') >= 0.02)
model.prod_E_ub_SUB_26 = Constraint(expr=model.Mixer_Level_E * mixer_SUB(model, 'SUB_26') + prod_E_SUB(model,'SUB_26') <= 0.2)
model.prod_E_lb_SUB_20 = Constraint(expr=model.Mixer_Level_E * mixer_SUB(model, 'SUB_20') + prod_E_SUB(model,'SUB_20') >= 0.001)
model.prod_E_lb_SUB_19 = Constraint(expr=model.Mixer_Level_E * mixer_SUB(model, 'SUB_19') + prod_E_SUB(model,'SUB_19') >= 0.0001)
model.prod_E_ub_SUB_19 = Constraint(expr=model.Mixer_Level_E * mixer_SUB(model, 'SUB_19') + prod_E_SUB(model,'SUB_19') <= 0.0009)
model.prod_E_lb_SUB_18 = Constraint(expr=model.Mixer_Level_E * mixer_SUB(model, 'SUB_18') + prod_E_SUB(model,'SUB_18') >= 0.0001)
model.prod_E_ub_SUB_18 = Constraint(expr=model.Mixer_Level_E * mixer_SUB(model, 'SUB_18') + prod_E_SUB(model,'SUB_18') <= 0.0065)
model.prod_E_ub_SUB_10 = Constraint(expr=model.Mixer_Level_E * mixer_SUB(model, 'SUB_10') + prod_E_SUB(model,'SUB_10') <= 0.029)
model.prod_E_lb_SUB_24 = Constraint(expr=model.Mixer_Level_E * mixer_SUB(model, 'SUB_24') + prod_E_SUB(model,'SUB_24') >= 0)
model.prod_E_ub_SUB_24 = Constraint(expr=model.Mixer_Level_E * mixer_SUB(model, 'SUB_24') + prod_E_SUB(model,'SUB_24') <= 0.2)
model.prod_E_lb_SUB_09 = Constraint(expr=model.Mixer_Level_E * mixer_SUB(model, 'SUB_09') + prod_E_SUB(model,'SUB_09') >= 0.002)
model.prod_E_ub_SUB_09 = Constraint(expr=model.Mixer_Level_E * mixer_SUB(model, 'SUB_09') + prod_E_SUB(model,'SUB_09') <= 0.05)
model.prod_E_lb_SUB_16 = Constraint(expr=model.Mixer_Level_E * mixer_SUB(model, 'SUB_16') + prod_E_SUB(model,'SUB_16') >= 0.0055)
model.prod_E_ub_SUB_16 = Constraint(expr=model.Mixer_Level_E * mixer_SUB(model, 'SUB_16') + prod_E_SUB(model,'SUB_16') <= 0.018)
model.prod_E_ub_SUB_15 = Constraint(expr=model.Mixer_Level_E * mixer_SUB(model, 'SUB_15') + prod_E_SUB(model,'SUB_15') <= 0.7)
model.prod_E_lb_SUB_23 = Constraint(expr=model.Mixer_Level_E * mixer_SUB(model, 'SUB_23') + prod_E_SUB(model,'SUB_23') >= 0)
model.prod_E_ub_SUB_23 = Constraint(expr=model.Mixer_Level_E * mixer_SUB(model, 'SUB_23') + prod_E_SUB(model,'SUB_23') <= 0.16)


# Define model_9 as an expression (Contaminant Model)
def model_9(model):
    total_SUB_25 = (model.Mixer_Level_A * mixer_SUB(model, 'SUB_25') + prod_A_SUB(model,'SUB_25')) + (model.Mixer_Level_B * mixer_SUB(model, 'SUB_25') + prod_B_SUB(model,'SUB_25')) + \
                   (model.Mixer_Level_C * mixer_SUB(model, 'SUB_25') + prod_C_SUB(model,'SUB_25')) + (model.Mixer_Level_D * mixer_SUB(model, 'SUB_25') + prod_D_SUB(model,'SUB_25')) + \
                   (model.Mixer_Level_E * mixer_SUB(model, 'SUB_25') + prod_E_SUB(model,'SUB_25'))
    return 400 * total_SUB_25

# model_9 constraint (Systemic Constraint for Contaminant)
model.model_9_constraint = Constraint(expr=model_9(model) <= 5.25)

# Small positive number to enforce strict inequality
epsilon = 0.001

# Constraint to ensure Mixer_Level_A is not equal to Mixer_Level_B
model.Mixer_Inequality1AB = Constraint(expr=abs (model.Mixer_Level_A - model.Mixer_Level_B) >= epsilon)
model.Mixer_Inequality1AC = Constraint(expr=abs (model.Mixer_Level_A - model.Mixer_Level_C) >= epsilon)
model.Mixer_Inequality1AE = Constraint(expr=abs (model.Mixer_Level_A - model.Mixer_Level_E) >= epsilon)
model.Mixer_Inequality2BC = Constraint(expr=abs (model.Mixer_Level_B - model.Mixer_Level_C) >= epsilon)
model.Mixer_Inequality2BD = Constraint(expr=abs (model.Mixer_Level_B - model.Mixer_Level_D) >= epsilon)
model.Mixer_Inequality3 = Constraint(expr=abs (model.Mixer_Level_C - model.Mixer_Level_D) >= epsilon)
model.Mixer_Inequality4 = Constraint(expr=abs (model.Mixer_Level_D - model.Mixer_Level_E) >= epsilon)

### OBJECTIVE FUNCTION FORMULA ###
def prod_A(model):
    return (model.Mixer_Level_A * model_6(model) + model_7A(model))
def prod_B(model):
    return (model.Mixer_Level_B * model_6(model) + model_7B(model))
def prod_C(model):
    return (model.Mixer_Level_C * model_6(model) + model_7C(model))
def prod_D(model):
    return (model.Mixer_Level_D * model_6(model) + model_7D(model))
def prod_E(model):
    return (model.Mixer_Level_E * model_6(model) + model_7E(model))

# Define Objective Function
def objective_function(model):
    return 0.0004 *(prod_A(model)+prod_B(model)+prod_C(model)+prod_D(model)+prod_E(model) )

model.objective = Objective(rule=objective_function, sense=minimize)

# Solver
solver = SolverFactory('ipopt',executable='/content/drive/MyDrive/ColabNotebooks/ipopt')
result = solver.solve(model, tee=True)
# # Solver
# solver = SolverFactory('ipopt')
# result = solver.solve(model, tee=True)

# Print Results
print("")
print("Objective Value = ", round(model.objective(), 3))
for rm in model.RM:
    print(f'Mixer [{rm}] = ', round(model.Mixer_RM_qty[rm](), 4))
print("")
print("Mixer Level Prod A = ", round(model.Mixer_Level_A(), 4))
print("")
for rm in model.RM:
    print(f'Prod_A [{rm}] = ', round(model.prod_A[rm](), 4))
print("")
print("Mixer Level Prod B = ", round(model.Mixer_Level_B(), 4))
print("")
for rm in model.RM:
    print(f'Prod_B [{rm}] = ', round(model.prod_B[rm](), 4))
print("")
print("Mixer Level Prod C = ", round(model.Mixer_Level_C(), 4))
print("")
for rm in model.RM:
    print(f'Prod_C [{rm}] = ', round(model.prod_C[rm](), 4))
print("")
print("Mixer Level Prod D = ", round(model.Mixer_Level_D(), 4))
print("")
for rm in model.RM:
    print(f'Prod_D [{rm}] = ', round(model.prod_D[rm](), 4))
print("")
print("Mixer Level Prod E = ", round(model.Mixer_Level_E(), 4))
print("")
for rm in model.RM:
    print(f'Prod_E [{rm}] = ', round(model.prod_E[rm](), 4))

### Extracting the Mixer RM quantities ####
mixer_quantities = {rm: model.Mixer_RM_qty[rm].value for rm in model.RM}

# Filter out zero values
non_zero_mixer_quantities = {rm: qty for rm, qty in mixer_quantities.items() if qty > 0.001}

# Generate a color map to ensure distinct colors for each slice
colors = cm.get_cmap('tab20').colors  # 'tab20' provides a palette of 20 distinct colors

# Plotting the pie chart for non-zero Mixer RM quantities only
plt.figure(figsize=(8, 6))

# Update the autopct to show up to 2 decimal places, and assign the color map
plt.pie(non_zero_mixer_quantities.values(), labels=non_zero_mixer_quantities.keys(),
        autopct='%.2f%%', startangle=140, colors=colors[:len(non_zero_mixer_quantities)])

plt.title('Compostion of RM in the Mixer')
plt.show()

### Extracting the Products RM quantities ###
# Assuming model, Prod_A to Prod_E are defined in the model and solved
products = ['prod_A', 'prod_B', 'prod_C', 'prod_D', 'prod_E']

# Define all RMs to ensure the color mapping covers all items
all_RMs = set()
for product in products:
    all_RMs.update(getattr(model, product).keys())

# Generate a color map
colors = cm.get_cmap('tab20', len(all_RMs)).colors  # Use 'tab20' with a sufficient number of colors

# Create a consistent color mapping for each RM
color_map = {rm: colors[i] for i, rm in enumerate(sorted(all_RMs))}

# Iterate over each product
for product in products:
    # Extract the product RM quantities dynamically using getattr
    Prod_RM = {rm: getattr(model, product)[rm].value for rm in model.RM}

    # Filter out zero values
    non_zero_Prod_qty = {rm: qty for rm, qty in Prod_RM.items() if qty > 0.001}

    # Calculate the total quantity for percentage calculation
    total_qty = sum(non_zero_Prod_qty.values())

    # Ensure the colors are applied consistently using the color_map
    item_colors = [color_map[rm] for rm in non_zero_Prod_qty.keys()]

    # Set the explode parameter to slightly separate small slices
    explode = [0.1 if qty < 0.05 else 0 for qty in non_zero_Prod_qty.values()]  # explode small slices

    # Plotting the pie chart for non-zero product RM quantities only
    plt.figure(figsize=(8,6))

    # Update the autopct to show up to 2 decimal places, and assign the consistent colors
    wedges, texts, autotexts = plt.pie(
        non_zero_Prod_qty.values(),
        labels=non_zero_Prod_qty.keys(),
        autopct='%.2f%%',
        startangle=140,
        colors=item_colors,  # Use the consistent colors based on the color_map
        explode=explode,  # Explode the small slices
        wedgeprops=dict(width=0.4, edgecolor='w'),  # Reduce the width to make the hole smaller
        textprops=dict(color="black", fontsize=10)  # Set text properties to black and size 10
    )

    # Improve label visibility by adjusting font size and weight
    for text in texts:
        text.set_fontsize(10)
        text.set_color('black')  # Ensure all labels are black
    for autotext in autotexts:
        autotext.set_fontsize(8)
        autotext.set_weight('bold')
        autotext.set_color('black')  # Percentage labels in black for consistency

    # Prepare legend labels with both names, absolute values, and percentages
    legend_labels = [
        f"{rm}: ({(qty/total_qty)*100:.2f}%)"
        for rm, qty in non_zero_Prod_qty.items()
    ]

    # Add a legend with values included, positioned to the right side
    plt.legend(legend_labels, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=10)

    # Set the title dynamically based on the current product
    plt.title(f'Product Composition of  {product.replace("_", " ").title()}')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # Display the plot
    plt.show()

### Extracting the mixer level values for each product ###
product_levels = {
    'Prod_A': model.Mixer_Level_A.value,
    'Prod_B': model.Mixer_Level_B.value,
    'Prod_C': model.Mixer_Level_C.value,
    'Prod_D': model.Mixer_Level_D.value,
    'Prod_E': model.Mixer_Level_E.value,
}

# Creating the bar chart
plt.figure(figsize=(10, 6))
bars = plt.bar(product_levels.keys(), product_levels.values(), color='skyblue')

# Adding titles and labels
plt.title('Mixer Levels for Each Product')
plt.xlabel('Products')
plt.ylabel('Mixer Level (%)')

# Annotating the bars with the values as percentages
for bar in bars:
    yval = bar.get_height() * 100  # Convert to percentage
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{yval:.2f}%', ha='center', va='bottom')

# Formatting the y-axis to show percentage
plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x * 100:.0f}%'))

# Displaying the bar chart
plt.show()

# Extract the components for each product and use value() to get the numeric values
sub25_A = 400 * (value(model.Mixer_Level_A) * value(mixer_SUB(model, 'SUB_25')) + value(prod_A_SUB(model, 'SUB_25')))
sub25_B = 400 * (value(model.Mixer_Level_B) * value(mixer_SUB(model, 'SUB_25')) + value(prod_B_SUB(model, 'SUB_25')))
sub25_C = 400 * (value(model.Mixer_Level_C) * value(mixer_SUB(model, 'SUB_25')) + value(prod_C_SUB(model, 'SUB_25')))
sub25_D = 400 * (value(model.Mixer_Level_D) * value(mixer_SUB(model, 'SUB_25')) + value(prod_D_SUB(model, 'SUB_25')))
sub25_E = 400 * (value(model.Mixer_Level_E) * value(mixer_SUB(model, 'SUB_25')) + value(prod_E_SUB(model, 'SUB_25')))

# Store these values in a dictionary for easier plotting
components = {
    'Sub25_A': sub25_A,
    'Sub25_B': sub25_B,
    'Sub25_C': sub25_C,
    'Sub25_D': sub25_D,
    'Sub25_E': sub25_E
}

# Create the stacked bar chart
plt.figure(figsize=(10, 6))

# Starting position for the bottom of the next bar segment
bottom = 0

# Plot each bar segment and annotate with its value
for key, value in components.items():
    plt.bar('SUB_25', value, bottom=bottom, label=key)
    plt.text('SUB_25', bottom + value / 2, f'{value:.2f}', ha='center', va='center', fontsize=10)
    bottom += value

# Set the upper limit for the y-axis
plt.ylim(top=7)

# Add a dotted line at y=5.25 to indicate the upper limit
plt.axhline(y=5.25, color='red', linestyle='--', linewidth=2, label='Upper Limit 5.25')

# Add labels and title
plt.title('Upper Bound of Systemic Constraint SUB_25')
plt.xlabel('SUB_25')
plt.ylabel('Value')
plt.legend()

# Display the chart
plt.show()

