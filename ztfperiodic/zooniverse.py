# Borrowing liberally from GravitySpy here
# https://github.com/Gravity-Spy/GravitySpy/blob/develop/gravityspy/api/project.py

from panoptes_client import Panoptes, Project, SubjectSet, Subject, Workflow, Classification
import pandas as pd
from pandas.io.json import json
import datetime
import math

# This function generically flatten a dict
def flatten(d, parent_key='', sep='_'):
    """Parameters
    ----------

    Returns
    -------
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        try:
            items.extend(flatten(v, new_key, sep=sep).items())
        except:
            items.append((new_key, v))
    return dict(items)

def workflow_with_most_answers(db):
    """Parameters
    ----------

    Returns
    -------
    """
    maxcount = max(len(v) for v in db.values())
    return [k for k, v in db.items() if len(v) == maxcount]

class ZooProject:
    def __init__(self, username='', password='',
                 project_id=None,
                 #workflow_order=[16000]):
                 workflow_order=None):
        self.username = username
        self.password = password
        self.project_id = project_id
        self.project = self.__connect()
        self.project_info = flatten(self.project.raw)

        # Determine workflow order
        self.workflow_info = {}
        if workflow_order is None:
            #print(sorted(self.project_info.keys()))
            order = self.project_info['links_active_workflows']
            #order = self.project_info['configuration_workflow_order']
        else:
            order = workflow_order

        workflows = [int(str(iWorkflow)) for iWorkflow in order]
        self.workflow_order = workflows

        # Save workflow information
        for iWorkflow in workflows:
            tmp1 = Workflow.find(iWorkflow)
            self.workflow_info[str(iWorkflow)] = flatten(tmp1.raw)

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

    def add_new_subject_timeseries(self, image_list, metadata_list, subject_set_name):
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

    def get_answers(self, workflow=None):
        """Parameters
        ----------
        workflow : `int`, optional, default None

        Returns
        -------
        A dict with keys of workflow IDs and values list
        of golden sets associated with that workflow
        """
        # now determine infrastructure of workflows so we know what workflow
        # this image belongs in
        workflowDictAnswers = {}

        if workflow:
            workflows = [str(workflow)]
        else:
            workflows = self.workflow_info.keys()

        # Determine possible answers to the workflows
        for iWorkflow in workflows:
            answerDict = {}
            try:
                answers = self.workflow_info[iWorkflow]['tasks_T1_choicesOrder']
            except:
                answers = self.workflow_info[iWorkflow]['tasks_T0_choicesOrder']

            for answer in answers:
                answerDict[answer] = []
            workflowDictAnswers[iWorkflow] = answerDict

        self.workflowDictAnswers = workflowDictAnswers
        return workflowDictAnswers


    def get_subject_sets_per_workflow(self, workflow=None):
        """Parameters
        ----------
        workflow : `int`, optional, default None

        Returns
        -------
        A dict with keys of workflow IDs and values list
        of golden sets associated with that workflow
        """
        workflowDictSubjectSets = {}
        workflowGoldenSetDict = self.get_golden_subject_sets()

        if workflow is not None:
            workflows = [str(workflow)]
        else:
            workflows = self.workflow_info.keys()

        for iWorkflow in workflows:
            # check if golden set exists for workflow
            goldenset = workflowGoldenSetDict[iWorkflow]
            # Determine subject sets associated with this workflow
            subject_sets_for_workflow = self.workflow_info[iWorkflow]\
                                        ['links_subject_sets']
            subjectset_id = [int(str(iSubject)) \
                            for iSubject in subject_sets_for_workflow]
            subjectset_id = [iSubject for iSubject in subjectset_id\
                            if iSubject not in goldenset]
            workflowDictSubjectSets[iWorkflow] = subjectset_id

        self.workflowDictSubjectSets = workflowDictSubjectSets
        return workflowDictSubjectSets

    def parse_classifications(self, last_id=None):
        """
        Parse classifications
        :return:
        """

        # Created empty list to store the previous classifications
        classificationsList = []
        
        # Query the last 100 classificaitons made (this is the max allowable)
        if not last_id is None:
            all_classifications = Classification.where(scope='project',
                                               project_id=self.project_id,
                                               last_id='{0}'.format(last_id),
                                               page_size='100')
        else:
            all_classifications = Classification.where(scope='project',
                                               project_id=self.project_id,
                                               page_size='100')
 
        # Loop until no more classifications
        for iN in range(0,all_classifications.object_count):
            try:
                classification = all_classifications.next()
            except:
                break
        
            # Generically with no hard coding we want to parse all aspects of the
            # classification metadata. This will ease any changes on the api side and
            # any changes to the metadata on our side.
        
            try:
                rawdata = flatten(classification.raw)
                rawdata['links_subjects'] = rawdata['links_subjects'][0]
                classificationsList.append(rawdata)
            except:
                continue

        if not classificationsList:
            raise ValueError('No New Classifications')
        
        # Now we want to make a panda data structure with all this information
        classifications = pd.DataFrame(classificationsList)
        annotations = classifications.annotations
        choices = []
        for annotation in annotations:
            try:
                choices.append(annotation[0]["value"][0]["choice"])
            except:
                choices.append('NOLABEL')
        choices = pd.DataFrame({'choices': choices})
        classifications = classifications.join(choices)

        classifications = classifications.apply(pd.to_numeric,
                                                axis=0, errors='ignore')
        classifications.created_at = pd.to_datetime(classifications.created_at,
                                                    infer_datetime_format=True)
        classifications.metadata_started_at = pd.to_datetime(classifications.metadata_started_at,infer_datetime_format=True)
        classifications.metadata_finished_at = pd.to_datetime(classifications.metadata_finished_at,infer_datetime_format=True)
        classifications = classifications.loc[classifications.choices!="NOLABEL"]
      
        classifications = classifications[['created_at','id','links_project','links_subjects','links_user','links_workflow','metadata_finished_at','metadata_started_at','metadata_workflow_version','choices']]
        classifications.loc[classifications.links_user.isnull(),'links_user'] = 0
        classifications.links_user = classifications.links_user.astype(int)
        classifications.loc[classifications.links_subjects.isnull(),'links_subjects'] = 0
        classifications.links_subjects = classifications.links_subjects.astype(int)
        classifications.choices = classifications.choices.astype(str)     
        return classifications 
