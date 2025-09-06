import sys
import pandas as pd

def process_new(baseline, n):
    data = baseline.groupby(['id2', 'id5']).aggregate(lambda x : tuple(x)).reset_index()
    data['score'] = data.apply(lambda x: [(off,pred) for off, pred in zip(x['id3'], x['pred'])], axis=1)
    data['depnt_var'] = data.apply(lambda x: [(off,depnt_var) for off, depnt_var in zip(x['id3'], x['y'])], axis=1)
    data['score_n'] = data['score'].map(lambda x: x[:n])
    data['depnt_var_n'] = data['depnt_var'].map(lambda x: x[:n])
    data['precision'] = data['depnt_var_n'].map(lambda x: sum([i[1] for i in x])/n)
    data['recall'] = data.apply(lambda x: 0 if sum([i[1] for i in x['depnt_var']])==0 else sum([i[1] for i in x['depnt_var_n']])/sum([i[1] for i in x['depnt_var']]), axis=1)
    data['temp2'] = data['depnt_var_n'].map(lambda x: len(x))
    data['prec_1'] = data['depnt_var_n'].map(lambda x: (sum([i[1] for i in x[:1]])/1)*x[0][1])
    data['prec_2'] = data['depnt_var_n'].map(lambda x: 0 if len(x)<2 else (sum([i[1] for i in x[:2]])/2)*x[1][1])
    data['prec_3'] = data['depnt_var_n'].map(lambda x: 0 if len(x)<3 else (sum([i[1] for i in x[:3]])/3)*x[2][1])
    data['prec_4'] = data['depnt_var_n'].map(lambda x: 0 if len(x)<4 else (sum([i[1] for i in x[:4]])/4)*x[3][1])
    data['prec_5'] = data['depnt_var_n'].map(lambda x: 0 if len(x)<5 else (sum([i[1] for i in x[:5]])/5)*x[4][1])
    data['prec_6'] = data['depnt_var_n'].map(lambda x: 0 if len(x)<6 else (sum([i[1] for i in x[:6]])/6)*x[5][1])
    data['prec_7'] = data['depnt_var_n'].map(lambda x: 0 if len(x)<7 else (sum([i[1] for i in x[:7]])/7)*x[6][1])
    data['MAP_num'] = data.apply(lambda x: (x['prec_1']+x['prec_2']+x['prec_3']+x['prec_4']+x['prec_5']+x['prec_6']+x['prec_7']), axis=1)
    data['MAP_den'] = data.apply(lambda x: min(sum([i[1] for i in x['depnt_var']]),n), axis=1)
    data['MAP'] = data.apply(lambda x: (x['prec_1']+x['prec_2']+x['prec_3']+x['prec_4']+x['prec_5']+x['prec_6']+x['prec_7'])/min(sum([i[1] for i in x['depnt_var']]),n), axis=1)
    data['depnt_vars_captured'] = data['depnt_var_n'].map(lambda x: sum([i[1] for i in x]))
    return(data['MAP'].mean())

score_file = sys.argv[1]
n = 7

baseline = pd.read_csv(score_file)
baseline['id5'] = pd.to_datetime(baseline['id5']).dt.date
baseline['id2'] = baseline['id2'].astype('str')
baseline['id3'] = baseline['id3'].astype('str')
assert (baseline['id2'].equals(baseline['id1'].apply(lambda x: str(x).split('_')[0]))), "id2 (customer) column issue"
assert (baseline['id3'].equals(baseline['id1'].apply(lambda x: str(x).split('_')[1]))), "id3 (offer) column issue"
assert (baseline['id5'].equals(pd.to_datetime(baseline['id1'].apply(lambda x: str(x).split('_')[-1])).dt.date)), "id5 (event date) column issue"
actual_r1 = pd.read_parquet("/axp/buanalytics/commtrshml/dev/asha19/CC_2025/r3/test_data_r3_y.parquet")
actual_r1['y'] = actual_r1['y'].astype('int')
actual_r1['id1'] = actual_r1['id1'].astype('str')

assert (baseline.pred.min() >= 0) & (baseline.pred.max() <= 1), 'prediction score should be in range [0,1]'
assert (baseline.shape[0] == 337714), f"Number of records not equal to 337714"
assert (baseline.shape[1] == 5), f"Number of columns not equal to 5"
assert set(['id1', 'pred', 'id2', 'id3', 'id5']).issubset(set(baseline.columns.tolist())), f"Incorrect columns parsed: Must contain - id1, pred, id2, id3, id5"

baseline_r1 = baseline.merge(actual_r1, how ='inner', on = 'id1')

merged_unique_ids = baseline_r1.id1.nunique()
unique_id1s = actual_r1.id1.nunique()
assert (merged_unique_ids == unique_id1s), 'id1 in submission does not match test data'

depnt_varers1 = baseline_r1.groupby(['id2', 'id5'])['y'].sum().reset_index()
depnt_varers1 = depnt_varers1[depnt_varers1.y>0][['id2', 'id5']]
baseline1 = baseline_r1.merge(depnt_varers1, on=['id2', 'id5'], how='inner')
viewers1 = baseline1.groupby(['id2', 'id5'])['y'].count().reset_index()
viewers1 = viewers1.rename({'y':'viewers_count'}, axis=1)
baseline1 = baseline1.merge(viewers1, on=['id2', 'id5'], how='inner')
baseline1 = baseline1.sort_values(['id2', 'id5', 'pred'], ascending=False)

b1 = process_new(baseline1.copy(), n)

print(b1)