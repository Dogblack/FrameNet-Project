try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import sys
import os
import pandas as pd


def parserelationFiles(c=0, relation_path='', filename='frRelation.xml', tagPrefix="{http://framenet.icsi.berkeley.edu}"):
    tree = ET.ElementTree(file=relation_path + filename)
    root = tree.getroot()

    fr_sub = []
    fr_sup = []
    fe_sub = []
    fe_sup = []
    fe_sub_to_frid = []
    fe_sup_to_frid = []
    fr_label =[]
    fe_label =[]
    rel_idx = -1
    rel_name = []

    for child in root.iter():
        if child.tag == tagPrefix + 'frameRelationType':
            rel_idx += 1
            rel_name.append(child.attrib.get('name'))

            for frame_child in child.iter():
                if frame_child.tag == tagPrefix + 'frameRelation':
                    fr_sub.append(int(frame_child.attrib.get('subID')))
                    fr_sup.append(int(frame_child.attrib.get('supID')))
                    fr_label.append(rel_idx)

                    for fe_child in frame_child.iter():
                        if fe_child.tag == tagPrefix + 'FERelation':
                            fe_sub.append(int(fe_child.attrib.get('subID')))
                            fe_sup.append(int(fe_child.attrib.get('supID')))
                            fe_sub_to_frid.append(int(frame_child.attrib.get('subID')))
                            fe_sup_to_frid.append(int(frame_child.attrib.get('supID')))
                            fe_label.append(rel_idx)

    return fr_sub, fr_sup, fe_sub, fe_sup, fr_label, fe_label, rel_name, fe_sub_to_frid, fe_sup_to_frid


# fr_sub, fr_sup, fe_sub, fe_sup, fr_label, fe_label, rel_name, fe_sub_to_frid, fe_sup_to_frid = parserelationFiles()
# print(fr_sub)
# print(fr_sup)
# print(fr_label)
# print(fe_sub)
# print(fe_sup)
# print(fe_label)
# print(fe_sub_to_frid)
# print(fe_sup_to_frid)
# print(rel_name)