import os
import pandas as pd


class Project:
    PROJECTS_FILE = 'projects.csv'

    def __init__(self,
                 project_info=None):
        if project_info is not None:
            self.project_info = project_info

    def save(self):
        project_name = self.project_info['project_name'][0]
        project_info = pd.DataFrame(self.project_info)

        if os.path.exists(self.PROJECTS_FILE):
            df = pd.read_csv(self.PROJECTS_FILE)
            df = df.loc[df['project_name'] != project_name]
            df = df.append(project_info)
            df.to_csv(self.PROJECTS_FILE, index=False)
        else:
            df = pd.DataFrame(project_info)
            df.to_csv(self.PROJECTS_FILE, index=False)

    def get_projects(self):
        return pd.read_csv(self.PROJECTS_FILE)

    def get(self, project_name):
        df = pd.read_csv(self.PROJECTS_FILE)
        return df.loc[df['project_name'] == project_name]
