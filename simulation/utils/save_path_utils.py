from os.path import join, isdir
from os import makedirs

def make_path_using_parameters(data):
    freq = data["freq"]
    lamda = data["lamda"]

    postfix_freq = "_".join(str(num) for num in freq)
    postfix_lambda = "_".join(str(num) for num in lamda)
    
    filename = f"exp_locled_lambda_{postfix_lambda}_freq_{postfix_freq}.csv"
    return filename 

def make_folder_and_path_name(data):
    depth = data["depth"][0] 
    betas = data["betas"][0]

    if len(betas) == 1 and depth == 2:
        folder = "single_element"
    elif len(betas) == 2 and depth == 2:
        folder = "tandem_element"
    elif len(betas) == 3 and depth == 2:     
        folder = "bifurcated"
    else:
        folder = "complex"

    path_name = join("results","comparison",folder)
    file_name = join(path_name,make_path_using_parameters(data))

    if isdir(f"{path_name}") == False: 
        makedirs(f"{path_name}",exist_ok=True)
  
    return file_name
