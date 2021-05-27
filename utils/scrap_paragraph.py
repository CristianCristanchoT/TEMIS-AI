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


# In[3]:


links_df = pd.read_csv("links_all.csv", index_col=0)


# In[4]:


links = links_df.LINK.values







# In[6]:

def scrap_paragraph(scrap_path, links):
    counter = 0
    for url in links:
        try:
            name = url.split("/")[-1].split(".")[0]
            if os.path.isfile(os.path.join(scrap_path, name + ".json")):
                counter += 1
                # viejo aviso, descomentar si se requiere
                # print("File {0} exists. Continue.".format(os.path.join(scrap_path, name + ".json")))
                continue

            print("{0} files detected. Skipping scrap.".format(counter))

            print("Processing link (" + str(url) + ")...")
            # diccionario que va a contener los párrafos del documento
            doc = {}
            url_request = requests.get(url)
            url_souped = BeautifulSoup(url_request.content, 'lxml')
            paragraphs = url_souped.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5'])
            p_list = []
            for i, p in enumerate(paragraphs):
                p_dict = {}
                if p.text.strip() == "":
                    continue
                p_dict["p_index"] = i
                p_dict["tag"] = p.name
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
            rand_sleep = random.uniform(0.8, 1.6)
            print("Safety sleep [{0}]...".format(rand_sleep))
            time.sleep(rand_sleep)
        except (TimeoutError, NewConnectionError, MaxRetryError, ConnectionError):
            with open("connection_error_links.txt", 'a') as f:
                f.write(url + "\n")
            print("Connection Error, link: {0}".format(url))
            print("Waiting 60 seconds for retry.")
            time.sleep(60)
        print("Finished.")


if __name__ == "__main__":
    links_df = pd.read_csv("links_all.csv", index_col=0)
    links = links_df.LINK.values
    scrap_path = "./scrappy/"
    scrap_paragraph(scrap_path, links)
