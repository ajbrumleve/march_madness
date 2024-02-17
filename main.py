import wx
import dill
from march_madness import *
import pandas as pd
import march_madness


class MainPanel(wx.Panel):

    def __init__(self, parent):
        super().__init__(parent)
        file = open("women_noseed_xgb_2023_march_madness.dill", "rb")
        self.data_instance = dill.load(file)
        file.close()
        # self.data_instance = march_madness.Data().load_model()
        self.data_instance.w_teams_pd = pd.read_csv('files/WTeams.csv')
        team1_label = wx.StaticText(self, label="First Team")
        team2_label = wx.StaticText(self, label="Second Team")
        self.team1 = wx.TextCtrl(self, style=wx.TE_PROCESS_ENTER)
        self.team2 = wx.TextCtrl(self, style=wx.TE_PROCESS_ENTER)
        button = wx.Button(self, label='Predict')
        self.result_txt = wx.TextCtrl(self, size=(500, 400))
        input_sizer1 = wx.BoxSizer(wx.VERTICAL)
        input_sizer2 = wx.BoxSizer(wx.VERTICAL)
        teams_sizer = wx.BoxSizer(wx.HORIZONTAL)
        input_sizer1.Add(team1_label, flag=wx.ALL | wx.CENTER, border=5)
        input_sizer1.Add(self.team1, flag=wx.ALL | wx.CENTER, border=5)
        input_sizer2.Add(team2_label, flag=wx.ALL | wx.CENTER, border=5)
        input_sizer2.Add(self.team2, flag=wx.ALL | wx.CENTER, border=5)
        teams_sizer.Add(input_sizer1)
        teams_sizer.Add(input_sizer2)
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        main_sizer.Add(teams_sizer, proportion=.5, flag=wx.ALL | wx.CENTER, border=5)
        main_sizer.Add(button, proportion=.5, flag=wx.ALL | wx.CENTER, border=5)
        main_sizer.Add(self.result_txt, proportion=.5, flag=wx.ALL | wx.CENTER, border=5)
        button.Bind(wx.EVT_BUTTON, self.predict)
        self.SetSizer(main_sizer)

    def predict(self,event):
        team_1_name = self.team1.GetValue()
        team_2_name = self.team2.GetValue()
        try:
            team1_vector = self.data_instance.getSeasonData(
                self.data_instance.teams_pd[self.data_instance.teams_pd['TeamName'] == team_1_name].values[0][0], 2022)
            team2_vector = self.data_instance.getSeasonData(
                self.data_instance.teams_pd[self.data_instance.teams_pd['TeamName'] == team_2_name].values[0][0], 2022)
        except:
            self.result_txt.SetValue("One of the schools may be misspelled. Please try again.")
            return

        self.result_txt.SetValue(
            f'Probability that {team_1_name} wins is {self.data_instance.predictGame(team1_vector, team2_vector, 0) * 100}%')


class MyFrame(wx.Frame):

    def __init__(self):
        super().__init__(None, title="2023 March Madness")
        panel = MainPanel(self)
        self.Show()


if __name__ == '__main__':
    march_madness
    app = wx.App(redirect=False)
    frame = MyFrame()
    app.MainLoop()
