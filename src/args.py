import argparse


def prepare_arg_parser():
    parser = argparse.ArgumentParser(prog="LoRA Test",
                                     description="Test LoRA")
    parser.add_argument("-ft",
                        "--fine-tune",
                        dest="do_fine_tune", 
                        help="",
                        action="store_true")
    
    parser.add_argument(
        "-i",
        "--infer",
        dest="do_inference",
        required=False, 
        help= "",
        type=str,
        choices=["base", "b", 
                 "fine_tuned", "ft"],
        nargs=1)
    
    parser.add_argument(
        "-p",
        "--prompt",
        dest="prompt",
        required=False, 
        help= "",
        type=str,
        nargs=1)
    
    parser.add_argument(
        "-m",
        "--model",
        dest="base",
        required=False, 
        help= "",
        type=str,
        nargs=1)


    parser.add_argument(
        "-g",
        "--generate-data",
        dest="do_generate_data",
        required=False, 
        help= "",
        action="store_true")
    return parser