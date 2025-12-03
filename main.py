# main.py
import os
import torch
from torchvision.utils import save_image
import config
from src import generator, analyzer, mosaic
from src.utils import log

def main():
    # 1. SETUP SUBFOLDER
    # Creates: output_samples/FISH/ or output_samples/ANIME/
    style_dir = os.path.join(config.OUTPUT_DIR, config.TILE_STYLE)
    os.makedirs(style_dir, exist_ok=True)
    
    log(f">> Output directory set to: {style_dir}")

    pipe = generator.load_sdxl()
    
    for i, prompt in enumerate(config.MACRO_PROMPTS):
        log(f"--- Scene {i+1} [{config.TILE_STYLE}] ---")
        
        # 1. Macro
        macro = generator.generate_macro(pipe, prompt)
        # Save macro to the subfolder
        save_image(macro, f"{style_dir}/scene_{i}_macro.png")
        
        # 2. Analyze
        palette_info, index_map = analyzer.quantize_macro(macro)
        
        # 3. Generate Supply
        palette_tiles = generator.generate_palette_tiles(pipe, palette_info)
        
        # 4. Assemble
        final = mosaic.assemble_mosaic(palette_tiles, index_map, macro)
        
        # Save final mosaic to the subfolder
        save_path = f"{style_dir}/scene_{i}_final_mosaic.png"
        save_image(final, save_path)
        log(f">> Saved to {save_path}")
        
        # Cleanup
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()