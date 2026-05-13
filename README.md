# Training Distribuito di un (Large) Language Model

## Installazione di GCloud CLI

Il primo passo consiste nell'[installare gcloud CLI](https://docs.cloud.google.com/sdk/docs/install-sdk?hl=it). Il programma di installazione di gcloud installer installerà Python v3.13 e i moduli di estensione richiesti per impostazione predefinita.

## Creare il cluster su GCloud

```bash
chmod +x ./setup-cluster.sh
./setup-cluster.sh
```

## Lanciare l'addestramento

```bash
chmod +x ./run-training.sh
./run-training.sh
```

A questo punto visitare l'indirizzo pubblico del master-node per monitorare i progressi dell'addestramento.
