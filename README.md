# Training Distribuito di un (Large) Language Model

## Installazione di GCloud CLI

Il primo passo consiste nell'[installare gcloud CLI](https://docs.cloud.google.com/sdk/docs/install-sdk?hl=it). Il programma di installazione di gcloud installer installerà Python v3.13 e i moduli di estensione richiesti per impostazione predefinita.

## Creare il cluster su GCloud

```bash
chmod +x setup-cluster.sh
./setup-cluster.sh
```

## Esercitazione 1

### Lanciare l'addestramento di un mini LLM

```bash
chmod +x train-llm.sh
./train-llm.sh
```

A questo punto visitare l'indirizzo pubblico del master-node per monitorare i progressi dell'addestramento.

## Esercitazione 2

Completare l'implementazione del codice presente nella cartella `smallimage/train.py` in modo che il task venga distribuito.

Lanciare i comandi:

```bash
chmod +x finetune-image.sh
./finetune-audio.sh
```

### Eseguire il fine tuning di un
