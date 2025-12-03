import torch
from torchvision.utils import save_image
import config
from src import generator, analyzer, mosaic
from src.utils import log

def main():
    pipe = generator.load_sdxl()
    
    for i, prompt in enumerate(config.MACRO_PROMPTS):
        log(f"--- Scene {i+1} ---")
        
        # 1. Macro
        macro = generator.generate_macro(pipe, prompt)
        save_image(macro, f"{config.OUTPUT_DIR}/scene_{i}_macro.png")
        
        # 2. Analyze (Quantize to 16 colors)
        palette_info, index_map = analyzer.quantize_macro(macro)
        
        # 3. Generate Supply (Only 16 images!)
        palette_tiles = generator.generate_palette_tiles(pipe, palette_info)
        
        # 4. Assemble
        final = mosaic.assemble_mosaic(palette_tiles, index_map, macro)        
        save_image(final, f"{config.OUTPUT_DIR}/scene_{i}_mosaic.png")
        log("Done.")
        
        # Cleanup
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()