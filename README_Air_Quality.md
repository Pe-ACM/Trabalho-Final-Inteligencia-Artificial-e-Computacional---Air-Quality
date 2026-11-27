# Classificação da qualidade do ar com o Air Quality UCI

Trabalho de classificação supervisionada baseado no dataset Air Quality da UCI.

## Target criado
`AirQualityClass` é derivado de `NO2(GT)`:
- 0: aceitável, NO2 <= 200 µg/m³
- 1: elevado, NO2 > 200 µg/m³

O corte de 200 µg/m³ corresponde ao valor-limite horário de NO2 apresentado pela Comissão Europeia. Neste trabalho, cada registro horário é classificado isoladamente; não se avalia a quantidade anual de excedências.

## Arquivos
- `AirQualityUCI.csv` e `AirQualityUCI.xlsx`: dados originais.
- `AirQualityUCI_preprocessed.csv`: dados tratados e target criado.
- `analise_air_quality.py`: pré-processamento, treinamento, avaliação e gráficos.
- `resultados_air_quality/`: métricas, relatórios, matrizes e gráficos.
- `Relatorio_Air_Quality_Pedro_Maurer.pdf`: relatório final.

## Modelos
Árvore de Decisão, Random Forest e KNN.

## Executar
```bash
pip install -r requirements_air_quality.txt
python analise_air_quality.py
```

## Fonte
VITO, Saverio. Air Quality. UCI Machine Learning Repository, 2008. DOI: 10.24432/C59K5F.
