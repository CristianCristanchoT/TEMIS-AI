#!/usr/bin/env python
# coding: utf-8

from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm 
from tqdm import trange
import json
import os
import time
import random

from urllib3.exceptions import TimeoutError, NewConnectionError, MaxRetryError, ConnectionError


def scrap(scrap_path, links):
    for url in links:
        name = url.split("/")[-1].split(".")[0]
        if os.path.isfile(os.path.join(scrap_path, name + ".json")):
            print("File {0} exists. Continue.".format(os.path.join(scrap_path, name + ".json")))
            continue

        print("Processing link (" + str(url) + ")...")
        # diccionario que va a contener los párrafos del documento
        doc = {}
        url_request = requests.get(url)
        url_souped = BeautifulSoup(url_request.content, 'lxml')
        paragraphs = url_souped.find_all('p')
        p_list = []
        for p in paragraphs:
            p_dict = {}
            if p.text.strip() == "":
                continue
            p_dict["text"] = p.text
            if "class" in p.attrs:
                p_dict["class"] = p.attrs["class"]
            else:
                p_dict["class"] = "no_class"
            p_list.append(p_dict)
        doc["name"] = name
        doc["paragraphs"] = p_list
        doc["link"] = url
        print("Saving as:", name + ".json")
        with open(os.path.join(scrap_path, name + ".json"), "w+") as f:
            json.dump(doc, f)
        rand_sleep = random.uniform(0.8, 2.5)
        print("Safety sleep [{0}]...".format(rand_sleep))
        time.sleep(rand_sleep)
    print("Finished.")



if __name__ == "__main__":
    links_df = pd.read_csv("links_all.csv", index_col=0)
    links = links_df.LINK.values
    scrap_path = "./scrap/"
    scrap(scrap_path, links)