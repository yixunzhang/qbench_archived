import xml.etree.ElementTree as ET
import numpy as np
import os

def get_file_names(xml_file):
    root = ET.parse(xml_file).getroot()
    names = []
    for node in root.findall("data"):
        names.append(node.attrib["target_file"])
    return names

def get_file_name_and_shape(xml_file):
    root = ET.parse(xml_file).getroot()
    name_and_shape = {}
    for node in root.findall("data"):
        name_and_shape[node.attrib["target_file"]] = eval(node.attrib["data_shape"])
    return name_and_shape

def check_dataset(xml_file):
    print(f"checking config file {xml_file}")
    names_and_shapes = get_file_name_and_shape(xml_file)
    for filename, shape in names_and_shapes.items():
        if not os.path.exists(filename):
            return False
        arr = np.load(filename, mmap_mode="r")
        if arr.shape != shape:
            return False
    return True

def generate_random_dataset(xml_file):
    print(f"generating data from config file {xml_file}")
    names_and_shapes = get_file_name_and_shape(xml_file)
    for filename, shape in names_and_shapes.items():
        print(f"Generating data with shape:{shape}")
        data = np.random.randn(*shape).astype(np.float32)
        np.save(filename, data)
        print(f"{filename} saved") 

if __name__== "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="generate random local data")
    parser.add_argument("-c", dest="xml_file", default=None, help="xml config file path")
    args = parser.parse_args()
    if args.xml_file is not None:
        generate_random_dataset(args.xml_file)
