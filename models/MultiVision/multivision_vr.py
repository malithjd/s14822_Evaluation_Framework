import pandas as pd
import json
import numpy as np
import itertools
import sys
import re
import altair as alt

import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
import torch.nn.functional as nnf

from model.encodingModel import ChartTypeNN, ChartTypeLSTM, ScoreNetLSTM
# from utils.helper import softmax, get_data_feature_by_column, get_embed_feature_by_column, get_all_charts_scores, charts_to_features
from utils.ChartRecommender import ChartRecommender
from utils.VegaLiteRender import VegaLiteRender


gpu = torch.device("cpu")

word_embedding_model_path = 'utils/en-50d-200000words.vec'

word_embedding_dict = {}
with open(word_embedding_model_path, encoding='utf-8') as file_in:
    lines = []
    for idx, line in enumerate(file_in):
        if idx == 0: ## line 0 is invalid
            continue
        word, *features = line.split()
        word_embedding_dict[word] = np.array(features)

column_score_model = ScoreNetLSTM(input_size=96, seq_length = 4, batch_size=2, pack = True).to(gpu)
column_score_model.load_state_dict(torch.load('trainedModel/singleChartModel.pt', map_location=gpu))
column_score_model.eval()

chart_type_model = ChartTypeLSTM(input_size = 96, hidden_size = 400, seq_length = 4, num_class = 9, bidirectional = True).to(gpu)
chart_type_model.load_state_dict(torch.load('trainedModel/chartType.pt', map_location=gpu))
chart_type_model.eval()

df = pd.read_csv('G:\\Internship\\Surge\\For Evaluation\\Project\\data\\preprocessed_tables\\1_Edu.csv')

## load model
mv_model = ScoreNetLSTM(input_size=9, seq_length=12).to(gpu)
mv_model.load_state_dict(torch.load('trainedModel/mvModel.pt', map_location=gpu))
mv_model.eval()

chartRecommender = ChartRecommender(df, word_embedding_dict, column_score_model, chart_type_model)

## Recommending an MV conditioned on current_mv
current_mv = [{'indices': (1,), 'chart_type': 'pie'}]
chartRecommender.recommend_mv(mv_model, current_mv = current_mv, max_charts = len(current_mv) + 1)

## Recommending an MV without conditions
recommended_charts_spec = chartRecommender.recommend_mv(mv_model, current_mv = [], max_charts = 4)

recommend_chart = recommended_charts_spec[3]
vr = VegaLiteRender(chart_type = recommend_chart['chart_type'], columns = recommend_chart['fields'], data = chartRecommender.df.to_dict('records'))
print(vr)

chart = alt.Chart.from_dict(vr.vSpec)
chart.save("MultiVisionChart1.json")


