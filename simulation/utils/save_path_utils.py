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
    
    base_path_name = join("results","comparison","complex_graph")

    file_name = join(base_path_name,make_path_using_parameters(data))

    if isdir(f"{base_path_name}") == False: 
        makedirs(f"{base_path_name}",exist_ok=True)
  
    return file_name
