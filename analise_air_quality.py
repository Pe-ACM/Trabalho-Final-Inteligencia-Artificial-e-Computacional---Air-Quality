import os, time, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

BASE=os.path.dirname(os.path.abspath(__file__))
OUT=os.path.join(BASE,'resultados_air_quality'); os.makedirs(OUT,exist_ok=True)
RAW=os.path.join(BASE,'AirQualityUCI.csv')

df=pd.read_csv(RAW,sep=';',decimal=',',encoding='utf-8')
df=df.dropna(axis=1,how='all').dropna(axis=0,how='all').copy()
df['datetime']=pd.to_datetime(df['Date']+' '+df['Time'],format='%d/%m/%Y %H.%M.%S',errors='coerce')
num=[c for c in df.columns if c not in ['Date','Time','datetime']]
df[num]=df[num].replace(-200,np.nan)
# Classification adaptation: elevated NO2 based on EU hourly reference of 200 ug/m3.
df=df[df['NO2(GT)'].notna() & df['datetime'].notna()].sort_values('datetime').reset_index(drop=True)
df['AirQualityClass']=(df['NO2(GT)']>200).astype(int)
features=['PT08.S1(CO)','PT08.S2(NMHC)','PT08.S3(NOx)','PT08.S4(NO2)','PT08.S5(O3)','T','RH','AH']
# Keep a reproducible preprocessed artifact; feature gaps remain NaN here and are imputed inside pipelines.
df[['datetime']+features+['NO2(GT)','AirQualityClass']].to_csv(os.path.join(BASE,'AirQualityUCI_preprocessed.csv'),index=False)
cut=int(len(df)*0.8)
train,test=df.iloc[:cut],df.iloc[cut:]
Xtr,ytr=train[features],train['AirQualityClass']; Xte,yte=test[features],test['AirQualityClass']
models={
 'Árvore de Decisão':Pipeline([('imputer',SimpleImputer(strategy='median')),('model',DecisionTreeClassifier(max_depth=6,min_samples_leaf=10,class_weight='balanced',random_state=42))]),
 'Random Forest':Pipeline([('imputer',SimpleImputer(strategy='median')),('model',RandomForestClassifier(n_estimators=300,max_depth=12,min_samples_leaf=3,class_weight='balanced',random_state=42,n_jobs=-1))]),
 'KNN':Pipeline([('imputer',SimpleImputer(strategy='median')),('scale',StandardScaler()),('model',KNeighborsClassifier(n_neighbors=9,weights='distance'))])
}
rows=[]; reports={}; cms={}
for name,m in models.items():
 t0=time.perf_counter(); m.fit(Xtr,ytr); train_s=time.perf_counter()-t0
 t0=time.perf_counter(); p=m.predict(Xte); pred_s=time.perf_counter()-t0
 rows.append({'Algoritmo':name,'Acurácia':accuracy_score(yte,p),'Precisão_macro':precision_score(yte,p,average='macro',zero_division=0),'Recall_macro':recall_score(yte,p,average='macro',zero_division=0),'F1_macro':f1_score(yte,p,average='macro',zero_division=0),'Tempo_treino_s':train_s,'Tempo_predicao_s':pred_s})
 reports[name]=classification_report(yte,p,target_names=['Aceitável (<=200)','Elevado (>200)'],output_dict=True,zero_division=0)
 cms[name]=confusion_matrix(yte,p).tolist()
res=pd.DataFrame(rows).sort_values('F1_macro',ascending=False); res.to_csv(os.path.join(OUT,'metricas_modelos.csv'),index=False)
json.dump(reports,open(os.path.join(OUT,'relatorios_por_classe.json'),'w',encoding='utf-8'),ensure_ascii=False,indent=2)
json.dump(cms,open(os.path.join(OUT,'matrizes_confusao.json'),'w',encoding='utf-8'),ensure_ascii=False,indent=2)
# Characterization before filtering target, after -200 -> missing
raw=pd.read_csv(RAW,sep=';',decimal=',',encoding='utf-8').dropna(axis=1,how='all').dropna(axis=0,how='all')
raw_num=[c for c in raw.columns if c not in ['Date','Time']]; clean=raw.copy(); clean[raw_num]=clean[raw_num].replace(-200,np.nan)
char=[]
for c in clean.columns:
 s=clean[c]
 if c in ['Date','Time']: interval=f'{s.dropna().iloc[0]} a {s.dropna().iloc[-1]}'; typ='data' if c=='Date' else 'hora'
 else: interval=f'{s.min():.4g} a {s.max():.4g}'; typ='numérico contínuo'
 char.append({'Atributo':c,'Tipo':typ,'Intervalo/valores válidos':interval,'Faltantes (-200)':int(s.isna().sum()),'Percentual_faltante':float(s.isna().mean())})
pd.DataFrame(char).to_csv(os.path.join(OUT,'caracterizacao_atributos.csv'),index=False)
# plots
counts=df['AirQualityClass'].value_counts().sort_index()
plt.figure(figsize=(6,4)); counts.plot(kind='bar',color=['#467599','#d9822b']); plt.xticks([0,1],['Aceitável','Elevado'],rotation=0); plt.ylabel('Registros'); plt.xlabel('Classe baseada em NO2'); plt.title('Distribuição da classe'); plt.tight_layout(); plt.savefig(os.path.join(OUT,'distribuicao_classe.png'),dpi=180); plt.close()
miss=clean.isna().sum().sort_values(ascending=False)
plt.figure(figsize=(8,4.5)); miss.plot(kind='bar',color='#8a5a44'); plt.ylabel('Valores faltantes'); plt.title('Valores faltantes por atributo'); plt.xticks(rotation=55,ha='right'); plt.tight_layout(); plt.savefig(os.path.join(OUT,'valores_faltantes.png'),dpi=180); plt.close()
ax=res.set_index('Algoritmo')[['Acurácia','Precisão_macro','Recall_macro','F1_macro']].plot(kind='bar',figsize=(8,4.8),ylim=(0,1),color=['#2f6690','#3a7d44','#d08b3e','#80475e']); ax.set_ylabel('Resultado'); ax.set_xlabel(''); ax.set_title('Comparação dos classificadores'); plt.xticks(rotation=0); plt.tight_layout(); plt.savefig(os.path.join(OUT,'comparacao_modelos.png'),dpi=180); plt.close()
# temporal label counts and summaries
summary={'raw_rows':len(clean),'model_rows':len(df),'train_rows':len(train),'test_rows':len(test),'date_min':str(df.datetime.min()),'date_max':str(df.datetime.max()),'class_counts_full':counts.to_dict(),'class_counts_test':yte.value_counts().sort_index().to_dict(),'split_datetime':str(test.datetime.iloc[0]),'most_missing':miss.index[0],'most_missing_count':int(miss.iloc[0])}
json.dump(summary,open(os.path.join(OUT,'resumo_dataset.json'),'w',encoding='utf-8'),ensure_ascii=False,indent=2)
print(json.dumps(summary,ensure_ascii=False,indent=2)); print(res.to_string(index=False))
for n in res.Algoritmo:
 print('\n',n,cms[n]); print(pd.DataFrame(reports[n]).T.iloc[:2][['precision','recall','f1-score','support']])
