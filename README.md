## SDXL-Based Recursive Stylized Mosaic Generation

This project generates stylized recursive photomosaics using Latent Diffusion Models. It constructs a macro image and procedurally generates micro-tiles to match the local color palette, creating a coherent multi-scale image.

### 1. Environment Setup

Install the required dependencies using pip:

```pip install -r requirements.txt```


### 2. Configuration

Open ```config.py``` to adjust generation parameters:

- **Set Tile Style**: Change ```TILE_STYLE``` to one of the supported modes:
    - "FLOWERS"
    - "ANIME"
    - "FISH"
- **Grid Size**: Adjust ```GRID_SIZE``` (Default: 64). Increasing this improves macro detail but increases generation time
- **Palette Size**: Adjust ```PALETTE_SIZE``` (Default: 64). Reducing this (e.g., to 16) speeds up the process significantly.

### 3. Execution

Run the main script to start the generation process.
Note: The first run will automatically download the SDXL model from Hugging Face.

```python main.py```
