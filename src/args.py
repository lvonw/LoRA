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
    return parser