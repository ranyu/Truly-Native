import cssutils as cu
import os, sys, logging, string, glob
from bs4 import BeautifulSoup as bs
import csv
import json


ferr = open("errors_in_scraping.log", "w")
PATH_TO_TRAIN_LABELS = "datasets/train.csv"


def map_of_data(PATH_TO_TRAIN_LABELS):
    """ Draw the map of id and sponsored from dataset
    parameters:
    --------------------------------------------------------
    PATH_TO_TRAIN_LABELS: path to the file of dataset
    """
    train_df = pd.read_csv(PATH_TO_TRAIN_LABELS)
    mapOfTrain = dict()
    
    for i, row in train_df.iterrows():
        row["id"] = str(row["file"].split('_')[0])
        mapOfTrain[row["id"]] = row["sponsored"]
        
    return mapOfTrain


def parse_page(in_file, urlid):
    """ Parse html files
    parameters:
    ---------------------------------------
    in_file: file to read raw_data from
    url_id: id of each page from file_name
    """
    page = open(in_file)
    soup = bs(page)

    content = soup.find_all(True)    # find all tags
    tags = []
    attrs = []
    values = []
    
    for i in content:
        name = i.name
        if name in ["html", "id"]:
            continue
        tags.append(name)

        for key, value in i.attrs.items():
            entry = str(name) + "_" + str(key)
            attrs.append(entry)
            if isinstance(value, list):
                values.append("list")
            elif isinstance(value, str) and len(value) < 15:
                values.append(value)
            
    return " ".join(tags), " ".join(attrs), " ".join(values)


def main(argv):
    """ This will loop over all raw_files and create processed ouput for
    a give site_id IF input data for that id exists.
    parameters:
    --------------------------------------------------------------------
    argv: sys args from the command line that consist of:
    <input_raw_dir> <output_file>
    * input_raw_dir: directory to read raw input html files
    * output_file: file to save processed html files
    """
    inFolder = argv[0]
    outputTrainFile = argv[1]
    outputTestFile = argv[2]
    traincsv = open(outputTrainFile, mode="w")
    feedstrain = csv.writer(traincsv)
    testcsv = open(outputTestFile, mode="w")
    feedstest = csv.writer(testcsv)
    mapOfTrain = map_of_data(PATH_TO_TRAIN_LABELS)

    cu.log.setLevel(logging.CRITICAL)
    fIn = glob.glob(inFolder + "/*/*raw*")
    feedstrain.writerow(["id", "tags", "attributes", "values", "sponsored"])
    feedstest.writerow(["id", "tags", "attributes", "values"])

    for idx, filename in enumerate(fIn):

        if idx % 1000 == 0:
            print "Processed %d HTML files" % idx
            if idx > 0:
                feedstrain.writerows(zip(train_id_array, train_tag_array, train_attr_array, train_value_array, y))
                feedstest.writerows(zip(test_id_array, test_tag_array, test_attr_array, test_value_array))
            y = []
            train_value_array = []
            train_attr_array = []
            train_tag_array = []
            train_id_array = []
            test_value_array = []
            test_attr_array = []
            test_tag_array = []
            test_id_array = []

        filenameDetails = filename.split("/")
        urlid = filenameDetails[-1].split('_')[0]

        try:
            tags, attrs, values = parse_page(filename, urlid)
        except Exception as e:
            ferr.write("parse error with reason : "+str(e)+" on page "+urlid+"\n")
            continue

        try:
            sponsored = mapOfTrain[urlid]
            y.append(sponsored)
            train_value_array.append(values)
            train_attr_array.append(attrs)
            train_tag_array.append(tags)
            train_id_array.append(urlid)
        except KeyError:
            test_value_array.append(values)
            test_attr_array.append(attrs)
            test_tag_array.append(tags)
            test_id_array.append(urlid)
        
    feedstrain.writerows(zip(train_id_array, train_tag_array, train_attr_array, train_value_array, y))
    feedstest.writerows(zip(test_id_array, test_tag_array, test_attr_array, test_value_array))
    traincsv.close()
    testcsv.close()
    ferr.close()
    

if __name__ == "__main__":
   main(["datasets/html", "datasets/trainTags.csv", "datasets/testTags.csv"])
