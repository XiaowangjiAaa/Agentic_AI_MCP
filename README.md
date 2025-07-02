# Agentic_AI_MCP

This repository contains a simple agentic system for crack image analysis.  
The code base has been reorganized to make each component easier to manage.

## Folder overview

- `agent` – memory, planner and other agent utilities
- `task_tools` – executable tools such as segmentation, quantification and summary advice
- `crack_metrics` – low level crack measurement functions
- `models` – neural network models (e.g. UNet)
- `utils` – helper utilities for paths, I/O and visualization
- `data` – sample images for testing

Run `python main_agent.py` to interact with the agent.
