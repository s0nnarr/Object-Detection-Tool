# Object-Detection-Tool

@ Stancu Luca - 2026 

This is the official AISight backend, suited for Runpod instances.


It's purpose is to:
    - stabilize a handshake with the user interface
    - decode incoming traffic via Websockets
    - verify incoming payloads via their declared size versus real size
    - optimize the .pt file into an .engine FP32 model file
    - run inference and return the results
