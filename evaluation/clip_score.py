from glob import glob
from T2IBenchmark.pipelines import calculate_clip_score

def load_prompts_as_list(txt_path):
    """
    Load prompts from a txt file, one per line.
    Returns a list of prompts indexed by line number.
    """
    with open(txt_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def clip_score(image_dir, prompt_txt_path):
    """
    Calculate the CLIP score using prompts from a txt file (indexed by line number).

    Parameters:
    image_dir (str): Directory containing the images (filenames should be numbered, e.g., 000001.png).
    prompt_txt_path (str): Path to txt file with one prompt per line.

    Returns:
    float: The calculated CLIP score.
    """
    try:
        cat_paths = sorted(glob(f"{image_dir}/*.png"))
        print(f"Number of image files: {len(cat_paths)}")

        prompts = load_prompts_as_list(prompt_txt_path)
        print(f"Loaded {len(prompts)} prompts.")

        # Build mapping: image_path -> corresponding prompt
        captions_mapping = {}
        for cat_path in cat_paths:
            filename = cat_path.split("/")[-1]
            index = int(filename.split(".")[0])  # assumes '000123.png' style
            if index < len(prompts):
                captions_mapping[cat_path] = prompts[index]
            else:
                print(f"Warning: No prompt for image {filename}")

        # Calculate CLIP score
        clip_score_value = calculate_clip_score(cat_paths, captions_mapping=captions_mapping, batch_size=16)

        return clip_score_value
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

