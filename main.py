""" Super Sense (#2C75FF)

Description:
    - main.py should be a ChatGPT-like interface:
        - Tab 1 is a chat interface, just as it looks on ChatGPT
        - Tab 2 is a video feed instead of chat history, and the predicted
           emotion will go in place where the input field usually goes
    - Qt is the UI framework used for this project

Folder structure (subject to change due to Qt):
    - datasets
        - camera
        - text
    - include
        - camera.py (function should be imported and return necessary output)
        - text.py (function should be imported and return necessary output)
        - trainer.py (allows for easier parameter tuning)
    - models
        - logs.txt (storing parameters used to camera given model)
        - model_c01 ('c' stands for 'camera' and then follows model revision)
        - model_c02
        - model_c03
        - ...
        - model_t01 ('t' stands for 'text' and then follows model revision)
        - model_t02
        - model_t03
        - ...
    - notebooks (metrics and important info about models in use)
        - camera.ipynb
        - text.ipynb
    - main.py (or future Qt executable, pending work)
    - README.md

Pending tasks:
    - Get final datasets ready
    - Prepare trainer.py for interaction with other Python files
    - Write documentation about models in their respective notebooks
    - Set up camera.py and text.py
    - Design the final UI to main.py

"""

print("Super Sense")