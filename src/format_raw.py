import os
import json

import constants






def main():
    save_path = os.path.join(constants.DATASETS_PATH_MASTER,
                         "Custom",
                         "SAP")

    data_path = os.path.join(save_path,
                            "raw_data.txt")
    
    with open(data_path, "r", encoding="utf-8") as file:
        content = file.read() 

    formatted_data = []
    for line in content.splitlines():
        cleaned_line = line.lstrip().rstrip()
        
        if not cleaned_line.startswith("{"):
            continue

        if not cleaned_line.endswith(","):
            if not cleaned_line.endswith("}"):
                cleaned_line += "}"
            cleaned_line += ","

        formatted_data.append(cleaned_line)


    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, 
                             "data.json")
    
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("[\n")
        for line in formatted_data:
            f.write(line + "\n") 
        f.write("]")


if __name__ == "__main__":
    main()