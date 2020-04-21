"""
Usage:
# Create train data:
python xml_to_csv.py -i [PATH_TO_IMAGES_FOLDER]/train -o [PATH_TO_ANNOTATIONS_FOLDER]/train_labels.csv

# Create test data:
python xml_to_csv.py -i [PATH_TO_IMAGES_FOLDER]/test -o [PATH_TO_ANNOTATIONS_FOLDER]/test_labels.csv
"""

import os
import glob
import pandas as pd
import argparse
import xml.etree.ElementTree as ET


def xml_to_csv(path):
    """Iterates through all .xml files (generated by labelImg) in a given directory and combines them in a single Pandas datagrame.

    Parameters:
    ----------
    path : {str}
        The path containing the .xml files
    Returns
    -------
    Pandas DataFrame
        The produced dataframe
    """

    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):   #walk into the directory
        tree = ET.parse(xml_file)                 #read each .xml file
        root = tree.getroot()                     #get root element
        for member in root.findall('object'):     #get the tag named 'object'(this tag contains bounding box co-ordinated)
            value = (root.find('filename').text,  #name of file, eg: 1training.jpg
                    int(root.find('size')[0].text),#width of image, eg:800 
                    int(root.find('size')[1].text),#height of image, eg: 600
                    member[0].text,                #name of the object to be detected, here 'carton'
                    int(member[4][0].text),        #xmin
                    int(member[4][1].text),        #ymin
                    int(member[4][2].text),        #xmax
                    int(member[4][3].text)         #ymax
                    )
            xml_list.append(value)                  # append all
    column_name = ['filename', 'width', 'height',
                'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name) # create pandas dataframe out of it
    return xml_df


def main():
    # Initiate argument parser
    parser = argparse.ArgumentParser(
        description="Sample TensorFlow XML-to-CSV converter")
    parser.add_argument("-i",
                        "--inputDir",
                        help="Path to the folder where the input .xml files are stored",
                        type=str)
    parser.add_argument("-o",
                        "--outputFile",
                        help="Name of output .csv file (including path)", type=str)
    args = parser.parse_args()

    if(args.inputDir is None):
        args.inputDir = os.getcwd()
    if(args.outputFile is None):
        args.outputFile = args.inputDir + "/labels.csv"

    assert(os.path.isdir(args.inputDir))

    xml_df = xml_to_csv(args.inputDir) # this will return a pandas dataframe
    xml_df.to_csv(
        args.outputFile, index=None)   #convert that dataframe to csv
    print('Successfully converted xml to csv.')


if __name__ == '__main__':
    main()
