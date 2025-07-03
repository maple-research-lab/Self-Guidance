import os
import torch
from accelerate import Accelerator
from tqdm import tqdm
import click 
import yaml
import dnnlib
from diffusers.utils import export_to_video

def CommandWithConfigFile(config_file_param_name):

    class CustomCommandClass(click.Command):

        def invoke(self, ctx):
            config_file = ctx.params[config_file_param_name]
            if config_file is not None:
                with open(config_file) as f:
                    config_data = yaml.load(f, Loader=yaml.FullLoader)
                    for key, value in config_data.items():
                        ctx.params[key] = value
            return super(CustomCommandClass, self).invoke(ctx)

    return CustomCommandClass

@click.command(cls=CommandWithConfigFile("config"))
@click.option("--config", type=str, default="config/flux.yaml")
@click.option('--local_data_path', type=str, default="VBench/prompts/prompts_per_dimension")
@click.option('--output', type=str, default="runs/flux/coco")
@click.option('--seed', type=int, default=0)

def main(
    config,
    local_data_path,
    output,
    seed,
    **kwargs,
):
    accelerator = Accelerator()
    device = accelerator.device

    network_kwargs = kwargs.pop("network_kwargs")
    pipe = dnnlib.util.construct_class_by_name(**network_kwargs)
    pipe.to(dtype=torch.bfloat16, device=device)

    dimension_list = ['human_action', 'scene','multiple_objects','appearance_style','overall_consistency']
    
    infer_kwargs = kwargs.pop("infer_kwargs")
    print("Using these configs:", infer_kwargs)
    output = os.path.join(output, f"cfg{infer_kwargs['cfg_scale']}-pag{infer_kwargs['pag_scale']}-sg{infer_kwargs['sg_scale']}_{infer_kwargs['sg_shift_scale']}_{infer_kwargs['sg_type']}_{infer_kwargs['sg_prev_max_t']}")
    os.makedirs(output, exist_ok=True)
    
    for dimension in dimension_list:
        # 读取 prompt 列表
        with open(f'{local_data_path}/{dimension}.txt', 'r') as f:
            all_prompts = [line.strip() for line in f.readlines()]

        # 分布式划分 prompt
        world_size = accelerator.num_processes
        rank = accelerator.process_index
        prompts_list = all_prompts[rank::world_size]

        # 创建保存目录
        cur_save_path = os.path.join(output, dimension)
        os.makedirs(cur_save_path, exist_ok=True)
    
        with torch.no_grad():
            for local_idx, prompt in enumerate(tqdm(prompts_list, disable=not accelerator.is_local_main_process)):
                for number in range(infer_kwargs['n_samples']):
                    save_file = os.path.join(cur_save_path, f"{prompt}-{number}.mp4")
                    if os.path.exists(save_file):
                        continue
                result = pipe(
                    prompt,
                    num_inference_steps=50,
                    height=480,
                    width=720,
                    guidance_scale=infer_kwargs['cfg_scale'],
                    pag_scale=infer_kwargs['pag_scale'],
                    generator=torch.manual_seed(seed),
                    self_guidance_scale=infer_kwargs['sg_scale'],
                    self_guidance_shift_t=infer_kwargs['sg_shift_scale'],
                    self_guidance_type=infer_kwargs['sg_type'],
                    sg_prev_max_t=infer_kwargs['sg_prev_max_t'],
                )
                video = result.frames[0]
                export_to_video(video, save_file, fps=8)

if __name__ == "__main__":
    main()

            
            
