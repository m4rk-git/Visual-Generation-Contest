import torch
from torchvision.utils import save_image
import config
from src import generator, analyzer, mosaic
from src.utils import log


def main():
    # 1. Load Model
    pipe = generator.load_sdxl()

    for i, prompt in enumerate(config.MACRO_PROMPTS):
        log(f"=========================================")
        log(f"Processing Scene {i+1}: {prompt}")
        log(f"=========================================")

        # 2. Generate Macro (The Target)
        macro_tensor = generator.generate_macro(pipe, prompt)
        save_image(macro_tensor, f"{config.OUTPUT_DIR}/scene_{i}_macro.png")

        # 3. Analyze Demand (The Optimization)
        # This tells us: "We need 400 blue tiles, 200 white tiles, 50 red tiles..."
        demand_plan = analyzer.analyze_demand(macro_tensor)

        # 4. Generate Supply (The Pool)
        # We generate exactly what is needed.
        pool_images, pool_feats = generator.generate_micro_pool(pipe, demand_plan)

        # 5. Build Mosaic
        final_mosaic = mosaic.build_mosaic(macro_tensor, pool_images, pool_feats)

        # 6. Save
        save_path = f"{config.OUTPUT_DIR}/scene_{i}_final_mosaic.png"
        save_image(final_mosaic, save_path)
        log(f">> Saved to {save_path}")

        # Optional: Clear pool to save memory or regenerate for next scene
        del pool_images
        del pool_feats
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
