from panoptes_client import Panoptes, Project, SubjectSet, Subject, Workflow
import pandas as pd
from pandas.io.json import json
import datetime
import math


class Subjects:
    def __init__(self, username='', password='', project_id=None):
        self.username = username
        self.password = password
        self.project_id = project_id
        self.project = self.__connect()

    def __connect(self):
        """
        Connect to the panoptes client api
        :return:
        """
        Panoptes.connect(username=self.username, password=self.password)
        return Project.find(self.project_id)

    def add_new_subject(self, image_list, metadata_list, subject_set_name):
        """
        Add a subject and the metadata.  image_list and metadata_list must be
        of equal length
        :param image_list: list of images to be added
        :param metadata_list: list of metadata to be added
        :return:
        """

        # Start by making sure we have two equal length list
        if len(image_list) != len(metadata_list):
            print("Image list and metadata list do not match")


        # Link to the subject set we want
        subject_set = SubjectSet()
        subject_set.links.project = self.project
        subject_set.display_name = subject_set_name
        subject_set.save()

        # Go through the image and metadata list and add the items
        new_subjects = []
        for i in range(len(image_list)):
            subject = Subject()
            subject.links.project = self.project
            subject.add_location(image_list[i])
            subject.metadata.update(metadata_list[i])
            subject.save()
            new_subjects.append(subject)

        subject_set.add(new_subjects)

    def remove_subject(self, subject_set_id, subject_list):
        """

        :param subject_list:
        :return:
        """
        subject_set = SubjectSet.find(subject_set_id)
        x = subject_set.remove(subject_list)
        y = subject_set.save()

        return x, y

    def link_new_set(self, subject_set_id):
        """

        :param subject_set_id:
        :return:
        """
        workflowSet = Workflow()
        subject = Subject()
        subject.links.project = self.project
        sset = subject.find(subject_set_id)

        print(1)
        workflowSet.links.project = self.project
        print(2)
        workflowSet.links.sub(sset)

    def subject_report_to_df(self, report):
        """

        :param report:
        :return:
        """
        # Create the dataframe
        df = pd.read_csv(report)

        # Change metadata column to dict from string
        #df = pd.concat([df, df['metadata'].apply(json.loads).apply(pd.Series)], axis=1).drop('metadata', axis=1)

        return df

    def convert_candid_to_value(self, candid, mdata=False):
        """

        :param candid:
        :return:
        """
        print(mdata, 'mdata')
        if mdata:
            if isinstance(candid, str):
                return -2

        if not isinstance(candid, str) and math.isnan(candid):
            return -1
        else:
            return int(candid)


    def convert_candid_to_value2(self, candid, mdata=True):
        """

        :param candid:
        :return:
        """

        if mdata:
            if isinstance(candid, str):
                return -2

        if not isinstance(candid, str) and math.isnan(candid):
            return -1
        else:
            return int(candid)

    def set_new_id(self, row):
        """

        :param row:
        :return:
        """
        if isinstance(row['Filename'], str) and 'zoo' in row['Filename']:
            return row['Filename'].replace('zoo', '').replace('.png', '')

        else:
            return int(row['candid'])

    def set_new_meta(self, row):

        value = str(row['newCandid'])
        end_code = value[-4:]


        if end_code == '5040' or end_code == '5168':
            # Run Like Query

            indices = [i for i, s in enumerate(img_list) if value[:-4] in s]
            if len(indices) == 1:
                return img_list[indices[0]].split('/')[-1].replace('.png\n','')[-18:]
            pass
        else:
            return row['newCandid']
            # Run Exact Query
            pass


    def match_subject_set_to_metadata(self, subject, metadata, start_date=None,
                                      end_date=None, parse_dates=True,
                                      outputfile=None):
        """

        :param subject:
        :param metadata:
        :param start_date:
        :param end_date:
        :param outputfile:
        :return:
        """

        df = self.subject_report_to_df(subject)

        if not df.empty and parse_dates:
            if not start_date and end_date:
                print("Input start_date and end_date needed")
                return
            elif start_date and not end_date:
                pass
#                end_date = (datetime.datetime.utcnow() + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
#                df = df[(df['created_at'] > start_date) & (df['created_at'] <= end_date)]

        # Now change the candid column in the metadata back to an int
        df['newMeta'] = df['newMeta'].apply(self.convert_candid_to_value)
        df['newMeta'] = df['newMeta'].apply(int)

        mdata_df = pd.read_csv(metadata)
        mdata_df['candid'] = mdata_df['candid'].apply(self.convert_candid_to_value2)
        mdata_df['candid'] = mdata_df['candid'].apply(int)
        print(df.keys())
        combine = pd.merge(df, mdata_df, left_on='newMeta', right_on='candid')

        if not outputfile:
            combine.to_csv('subject_metadata_merge2.csv')
        else:
            combine.to_csv(outputfile)

        return combine


