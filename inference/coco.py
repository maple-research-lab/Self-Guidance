import os
import torch
from accelerate import Accelerator
from tqdm import tqdm
import click 
import yaml
import dnnlib
from diffusers import EulerDiscreteScheduler

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
@click.option('--local_data_path', type=str, default="coco_val5000_prompts.txt")
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
    network_kwargs['torch_dtype'] = torch.float16 if network_kwargs['torch_dtype'] == "float16" else torch.bfloat16
    pipe = dnnlib.util.construct_class_by_name(**network_kwargs)
    pipe.to(device=device)

    # 读取 prompts
    with open(local_data_path, 'r', encoding='utf-8') as f:
        all_prompts = [line.strip() for line in f.readlines()]

    # 平均分配 prompts
    world_size = accelerator.num_processes
    rank = accelerator.process_index
    prompts_list = all_prompts[rank::world_size]  # 每个进程处理间隔为 world_size 的部分

    infer_kwargs = kwargs.pop("infer_kwargs")
    print("Using these configs:", infer_kwargs)

    scheduler_type = infer_kwargs.pop("scheduler_type", None)
    if scheduler_type == "euler":
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        
    output = os.path.join(output, f"cfg{infer_kwargs['cfg_scale']}-pag{infer_kwargs['pag_scale']}-sg{infer_kwargs['sg_scale']}_{infer_kwargs['sg_shift_scale']}_{infer_kwargs['sg_type']}_{infer_kwargs['sg_prev_max_t']}")
    os.makedirs(output, exist_ok=True)

    with torch.no_grad():
        for local_idx, prompt in enumerate(tqdm(prompts_list, disable=not accelerator.is_local_main_process)):
            global_idx = rank + local_idx * world_size
            save_path = os.path.join(output, f"{global_idx:04d}.png")
            if os.path.exists(save_path):
                continue
            images = pipe(
                prompt,
                guidance_scale=infer_kwargs['cfg_scale'],
                pag_scale=infer_kwargs['pag_scale'],
                generator=torch.manual_seed(seed),
                self_guidance_scale=infer_kwargs['sg_scale'],
                self_guidance_shift_t=infer_kwargs['sg_shift_scale'],
                self_guidance_type=infer_kwargs['sg_type'],
                sg_prev_max_t=infer_kwargs['sg_prev_max_t'],
            ).images

            # 计算全局 index
            
            images[0].save(save_path)

if __name__ == "__main__":
    main()

            
            
