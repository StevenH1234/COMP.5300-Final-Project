# import json
# import os
# import argparse
# import torch
# from PIL import Image
# from transformers import TextStreamer

# from colongpt.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
# from colongpt.conversation import conv_templates, SeparatorStyle
# from colongpt.model.builder import load_pretrained_model
# from colongpt.util.utils import disable_torch_init
# from colongpt.util.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria


# # for colon300k image files
# def load_image_from_json(json_file, image_dir, i):
#     with open(json_file, 'r') as f:
#         data_list = json.load(f)
#     image_id = data_list[i]['id']
#     image_file = os.path.join(image_dir, image_id)
#     # ----------- added -----------------
#     try:
#         image = Image.open(image_file).convert('RGB')
#     except FileNotFoundError:
#         return None 
#     # ----------- added -----------------
#     return image

# # inference: cli.py
# def main(args):
#     disable_torch_init()
#     model_name = get_model_name_from_path(args.model_path)
#     tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name,
#                                                                            args.model_type, args.load_8bit,
#                                                                            args.load_4bit, device=args.device)

#     conv_mode = "colongpt"

#     if args.conv_mode is not None and conv_mode != args.conv_mode:
#         print(
#             '[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode,
#                                                                                                               args.conv_mode,
#                                                                                                               args.conv_mode))
#     else:
#         args.conv_mode = conv_mode

#     with open(args.json_file, 'r') as f:
#         data_list = json.load(f)

#     if os.path.exists(args.output_path):
#         with open(args.output_path, 'r') as f:
#             predicted_data = json.load(f)
#         predicted_ids = [item['id'] for item in predicted_data]
#     else:
#         predicted_ids = []


#     # with open(args.output_path, 'a') as f:
#     #     f.write('[')
#     # if the file doesn’t exist or is empty, write the opening '['
#     first_item = True
#     # if not os.path.exists(args.output_path) or os.path.getsize(args.output_path) == 0:
#     # if the file already exists and is non-empty, remove its trailing ']' so we can append
#     if os.path.exists(args.output_path) and os.path.getsize(args.output_path) > 0:
#         with open(args.output_path, 'rb+') as f:
#             f.seek(-1, os.SEEK_END)
#             last = f.read(1)
#             if last == b']':
#                 # chop off that closing bracket
#                 f.truncate()
#     else:            
#         with open(args.output_path, 'a') as f:
#             f.write('[')

#     for i in range(len(data_list)):
#         item = data_list[i]
#         image_id = item['id']

#         if image_id in predicted_ids:
#             continue

#         conversations = item['conversations']

#         image = load_image_from_json(args.json_file, args.image_dir, i)
#         # ---------------- added -------------------------
#         if image == None:
#             continue
#         # ------------------------------------------------

#         # Similar operation in model_worker.py
#         image_tensor = process_images([image], image_processor, model.config)
#         if type(image_tensor) is list:
#             image_tensor = [image.to(model.device, dtype=model.dtype) for image in image_tensor]
#         else:
#             image_tensor = image_tensor.to(model.device, dtype=model.dtype)

#         # init new conversation
#         conv = conv_templates[args.conv_mode].copy()
#         roles = conv.roles

#         # use human `value` as prompt
#         conv.append_message(conv.roles[0], conversations[0]['value'])
#         conv.append_message(conv.roles[1], None)
#         prompt = conv.get_prompt()

#         input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(
#             model.device)
#         stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
#         keywords = [stop_str]
#         stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
#         streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

#         with torch.inference_mode():
#             output_ids = model.generate(
#                 input_ids,
#                 images=image_tensor,
#                 do_sample=True if args.temperature > 0 else False,
#                 temperature=args.temperature,
#                 max_new_tokens=args.max_new_tokens,
#                 streamer=streamer,
#                 use_cache=True,
#                 stopping_criteria=[stopping_criteria])

#         outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).replace("<|endoftext|>", "").strip()

#         conversations.append({
#             "from": "prediction",
#             "value": outputs
#         })

#         # with open(args.output_path, 'a') as f:
#         #     json.dump({
#         #         "id": image_id,
#         #         "image": item['image'],
#         #         "conversations": conversations
#         #     }, f, indent=4)
#         #     f.write(',')

#         with open(args.output_path, 'a') as f:
#             # if this isn’t the first object, prefix with a comma+newline
#             if not first_item:
#                 f.write(',\n')
#             json.dump({
#                 "id": image_id,
#                 "image": item['image'],
#                 "conversations": conversations
#             }, f, indent=4)
#             first_item = False

#     # with open(args.output_path, 'rb+') as f:
#     #     f.seek(-1, os.SEEK_END)
#     #     f.truncate()
#     # with open(args.output_path, 'a') as f:
#     #     f.write(']')
#     # write the closing bracket—if no items were written, this yields "[]"
#     with open(args.output_path, 'a') as f:
#         f.write(']')


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model_path", type=str, default=None)
#     parser.add_argument("--model_base", type=str, default=None)
#     parser.add_argument("--model_type", type=str, default=None)
#     parser.add_argument("--device", type=str, default="cuda")
#     parser.add_argument("--temperature", type=float, default=0)
#     parser.add_argument("--max_new_tokens", type=int, default=40)
#     parser.add_argument("--load-8bit", action="store_true")
#     parser.add_argument("--load-4bit", action="store_true")
#     parser.add_argument("--debug", action="store_true")
#     parser.add_argument("--json_file", type=str)
#     parser.add_argument("--image_dir", type=str)
#     parser.add_argument("--output_path", type=str)
#     parser.add_argument("--conv_mode", type=str, default=None)
#     args = parser.parse_args()
#     main(args)

# -------------------------------------------------------------

import json
import os
import argparse
import torch
from PIL import Image
from transformers import TextStreamer

from colongpt.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from colongpt.conversation import conv_templates, SeparatorStyle
from colongpt.model.builder import load_pretrained_model
from colongpt.util.utils import disable_torch_init
from colongpt.util.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)

def load_image_from_json(json_file, image_dir, i):
    """Load the i-th image by reading its ID from the JSON and opening it."""
    with open(json_file, 'r') as f:
        data_list = json.load(f)
    image_id = data_list[i]['id']
    image_file = os.path.join(image_dir, image_id)
    return Image.open(image_file).convert('RGB')

def main(args):
    # 1) Initialize model & tokenizer
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path,
        args.model_base,
        model_name,
        args.model_type,
        args.load_8bit,
        args.load_4bit,
        device=args.device
    )

    # Force the “colongpt” conversation template
    args.conv_mode = "colongpt"

    # 2) Load any existing predictions
    try:
        with open(args.output_path, 'r') as f:
            results = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        results = []
    existing_ids = {entry['id'] for entry in results}

    # 3) Read the input data list
    with open(args.json_file, 'r') as f:
        data_list = json.load(f)

    # 4) Inference loop
    for i, item in enumerate(data_list):
        image_id = item['id']
        if image_id in existing_ids:
            continue

        # Load and preprocess image
        try:
            image = load_image_from_json(args.json_file, args.image_dir, i)
        except FileNotFoundError:
            continue
        image_tensor = process_images([image], image_processor, model.config)
        if isinstance(image_tensor, list):
            image_tensor = [t.to(model.device, dtype=model.dtype) for t in image_tensor]
        else:
            image_tensor = image_tensor.to(model.device, dtype=model.dtype)

        # Build prompt
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], item['conversations'][0]['value'])
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX,
            return_tensors='pt'
        ).unsqueeze(0).to(model.device)

        # Stopping criteria & streamer
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        # Generate
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=(args.temperature > 0),
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

        outputs = tokenizer.decode(
            output_ids[0, input_ids.shape[1]:]
        ).replace("<|endoftext|>", "").strip()

        # Record the new prediction
        new_entry = {
            "id": image_id,
            "image": item["image"],
            "conversations": item["conversations"] + [
                {"from": "prediction", "value": outputs}
            ]
        }
        results.append(new_entry)
        existing_ids.add(image_id)

    # 5) Overwrite the output file with the complete list
    with open(args.output_path, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_base", type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=40)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--json_file", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--conv_mode", type=str, default=None)
    args = parser.parse_args()
    main(args)
