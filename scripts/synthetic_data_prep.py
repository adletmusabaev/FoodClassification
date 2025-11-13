"""
Synthetic Data Generation Helper for Food Classification
Uses TA-TiTok + MaskGen concept to generate diverse food images
"""

import json
import os
from datetime import datetime

class FoodSyntheticDataGenerator:
    """Helper class to generate prompts for synthetic food images"""
    
    def __init__(self, categories):
        self.categories = categories
        self.prompts = []
        
        # Lighting variations
        self.lighting = [
            "natural daylight", "warm indoor lighting", "bright restaurant lighting",
            "soft morning light", "evening golden hour", "overhead lighting",
            "side lighting", "backlit", "studio lighting"
        ]
        
        # Plating styles
        self.plating = [
            "white ceramic plate", "wooden board", "black slate plate",
            "colorful plate", "bamboo plate", "metal tray", "glass plate",
            "rustic wooden table", "marble countertop", "traditional dish"
        ]
        
        # Camera angles
        self.angles = [
            "top-down view", "45-degree angle", "side view", "close-up shot",
            "overhead shot", "slightly angled", "eye-level view", "bird's eye view"
        ]
        
        # Contexts
        self.contexts = [
            "on restaurant table", "home kitchen setting", "food photography setup",
            "cafe environment", "outdoor picnic", "buffet display", "street food stall",
            "fine dining presentation", "casual dining", "takeout container"
        ]
        
        # Food-specific modifiers
        self.food_modifiers = {
            "Bread": [
                "freshly baked", "sliced", "whole loaf", "artisan style",
                "crusty texture", "soft and fluffy", "toasted", "with butter"
            ],
            "Fried food": [
                "golden crispy", "freshly fried", "with dipping sauce",
                "on paper liner", "hot and steaming", "crunchy texture",
                "with garnish", "assorted pieces"
            ],
            "Seafood": [
                "fresh catch", "grilled", "raw sashimi style", "cooked and plated",
                "with lemon wedge", "on ice", "garnished", "steamed"
            ],
            "Vegetable-Fruit": [
                "fresh and colorful", "mixed variety", "arranged artistically",
                "washed and clean", "organic", "sliced", "whole pieces", "in basket"
            ]
        }
    
    def generate_prompts(self, category, num_prompts=10):
        """Generate diverse prompts for a specific food category"""
        
        prompts_list = []
        
        if category not in self.food_modifiers:
            print(f"Warning: Category '{category}' not recognized")
            return prompts_list
        
        modifiers = self.food_modifiers[category]
        
        for i in range(num_prompts):
            # Rotate through variations
            modifier = modifiers[i % len(modifiers)]
            lighting = self.lighting[i % len(self.lighting)]
            plate = self.plating[i % len(self.plating)]
            angle = self.angles[i % len(self.angles)]
            context = self.contexts[i % len(self.contexts)]
            
            # Create prompt
            prompt = (
                f"A high-quality food photograph of {modifier} {category.lower()}, "
                f"presented on {plate}, {angle}, {lighting}, {context}, "
                f"professional food photography, appetizing, detailed texture"
            )
            
            prompts_list.append({
                "category": category,
                "prompt": prompt,
                "settings": {
                    "modifier": modifier,
                    "lighting": lighting,
                    "plating": plate,
                    "angle": angle,
                    "context": context
                }
            })
        
        return prompts_list
    
    def generate_all_prompts(self, images_per_category=20):
        """Generate prompts for all categories"""
        
        all_prompts = {}
        
        for category in self.categories:
            print(f"\nGenerating {images_per_category} prompts for '{category}'...")
            prompts = self.generate_prompts(category, images_per_category)
            all_prompts[category] = prompts
            
            # Print sample
            if prompts:
                print(f"Sample prompt: {prompts[0]['prompt'][:100]}...")
        
        return all_prompts
    
    def save_prompts(self, prompts, output_dir="synthetic_prompts"):
        """Save prompts to JSON file"""
        
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_dir, f"prompts_{timestamp}.json")
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(prompts, f, indent=2, ensure_ascii=False)
        
        print(f"\n Prompts saved to: {filename}")
        return filename
    
    def create_generation_guide(self, prompts, output_dir="synthetic_prompts"):
        """Create a markdown guide for generating images"""
        
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_dir, f"generation_guide_{timestamp}.md")
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("# Synthetic Data Generation Guide\n\n")
            f.write("## How to Generate Images\n\n")
            f.write("### Option 1: Using Stable Diffusion (Recommended)\n")
            f.write("1. Use Stable Diffusion XL or similar model\n")
            f.write("2. Settings: 512x512 or 1024x1024, CFG Scale: 7-9, Steps: 30-50\n")
            f.write("3. Add negative prompt: 'blurry, low quality, watermark, text, distorted'\n\n")
            
            f.write("### Option 2: Using Online Services\n")
            f.write("- Leonardo.ai (free tier available)\n")
            f.write("- Midjourney\n")
            f.write("- DALL-E 3\n\n")
            
            f.write("### Option 3: Using Google Colab (Free)\n")
            f.write("- Use Stable Diffusion WebUI in Colab\n")
            f.write("- Notebook: https://colab.research.google.com/github/camenduru/stable-diffusion-webui-colab\n\n")
            
            f.write("## Generated Prompts by Category\n\n")
            
            for category, prompt_list in prompts.items():
                f.write(f"### {category} ({len(prompt_list)} prompts)\n\n")
                
                for idx, p in enumerate(prompt_list[:5], 1):  # Show first 5
                    f.write(f"{idx}. {p['prompt']}\n\n")
                
                if len(prompt_list) > 5:
                    f.write(f"... and {len(prompt_list) - 5} more prompts (see JSON file)\n\n")
        
        print(f" Generation guide saved to: {filename}")
        return filename


def main():
    """Main function to generate synthetic data prompts"""
    
    # Your food categories
    categories = ["Bread", "Fried food", "Seafood", "Vegetable-Fruit"]
    
    print("=" * 60)
    print("SYNTHETIC DATA PROMPT GENERATOR")
    print("=" * 60)
    
    # Initialize generator
    generator = FoodSyntheticDataGenerator(categories)
    
    # Generate prompts (adjust number based on your needs)
    images_per_category = 30  # Generate 30 prompts per category
    
    print(f"\nGenerating {images_per_category} prompts per category...")
    all_prompts = generator.generate_all_prompts(images_per_category)
    
    # Save prompts
    prompts_file = generator.save_prompts(all_prompts)
    guide_file = generator.create_generation_guide(all_prompts)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total_prompts = sum(len(prompts) for prompts in all_prompts.values())
    print(f"Total prompts generated: {total_prompts}")
    print(f"Categories: {len(categories)}")
    print(f"Prompts per category: {images_per_category}")
    
    print("\n Next steps:")
    print("1. Review the generated prompts in the JSON file")
    print("2. Use the generation guide (MD file) to generate images")
    print("3. Generate images using Stable Diffusion or similar tools")
    print("4. Save generated images in a 'synthetic' subfolder")
    print("5. Label all synthetic images with watermark/metadata")
    
    print("\n IMPORTANT:")
    print("- Label all synthetic images clearly")
    print("- Keep track of which images are synthetic vs real")
    print("- Add watermark: 'AI Generated' to synthetic images")
    print("- Document the generation process in your report")


if __name__ == "__main__":
    main()