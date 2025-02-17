import logging
import torch
import util

from transformers   import pipeline

def run_inference(inference_config,
                  model,
                  tokenizer,
                  prompt,
                  logger = logging,
                  cache_dir = None):
    
    response = None

    do_generate = True
    do_pipeline = False

    if do_generate:
        tokenized_input = tokenizer(prompt, return_tensors="pt")
        input_ids       = tokenized_input.input_ids
        attention_mask  = tokenized_input.attention_mask


        device = util.get_device(idle=False)

        model           = model.to(device) 
        input_ids       = input_ids.to(device)
        attention_mask  = attention_mask.to(device)
        logger.info (f"Moved model to {device}")


        output_sequences = model.generate(
            input_ids,
            max_new_tokens          = 2048, 
            attention_mask          = attention_mask,
            do_sample               = False, 
            temperature             = None if True else 0.6,
            top_k                   = None if True else 10,
            top_p                   = None if True else 0.9,
            num_return_sequences    = 1,
            pad_token_id            = tokenizer.eos_token_id,
            eos_token_id            = tokenizer.eos_token_id)

        output_sequences = output_sequences.to(
            util.get_device(idle = True))

        output_sequences = output_sequences[:, input_ids.shape[1]:]

        response = tokenizer.decode(
            output_sequences[0], 
            skip_special_tokens=inference_config["skip special tokens"])

    elif do_pipeline:
        inference_pipeline = pipeline(
            task            = "text-generation",
            model           = model,
            torch_dtype     = torch.bfloat16,
            device_map      = "auto",
            tokenizer       = tokenizer,
            model_kwargs    = {"cache_dir": cache_dir}
        )

        response = inference_pipeline(
            prompt,        
            do_sample               = True,
            top_k                   = 10,
            num_return_sequences    = 1,
            pad_token_id            = tokenizer.eos_token_id,
            bos_token_id            = tokenizer.bos_token_id,
            eos_token_id            = tokenizer.eos_token_id,
            truncation              = True,
            max_new_tokens          = 100,
        )   
    else:
        logging.warning("No valid inference method has been selected")


    return response