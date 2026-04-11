
Eventually I will structure the demo modules as follows:

demos/
├── app.py
├── theme.py
├── plots/
│   ├── streamflow.py      # shared streamflow comparison
│   ├── shm_states.py      # SHM internal states
│   └── dcfe_states.py     # dCFE internal states
└── pages/
    ├── streamflow.py      # main page, streamflow plots for all models
    └── parameters.py      # future parameter explorer

Streamflow predictions for all models will be located on streamflow.py page

parameters.py page will allow the user to select the hybrid model and explore the internal states and parameters associated with that specific model.
- dCFE and SHM both have different internal states and parameters.