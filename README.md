# From Cones Detection to Road Segmentation

## Description

This project uses various Python packages to manage YOLO models, process images and videos, and apply analysis and drawing techniques to images. Below are the instructions for setting up the required environment.

## System Requirements

Make sure you have installed:

    - Python 3.8 or higher
    - pip (Python package manager)

## Installing Requirements

To run this project, install the Python dependencies listed in requirements.txt using the following command:

```bash
pip install -r requirements.txt
```

### Contents of "requirements.txt"

Here are the libraries required for the project:

```
torch
ultralytics
Pillow
opencv-python
numpy
```

### Descrizione dei Moduli

- **torch**: Libreria per il calcolo scientifico e il deep learning.
- **ultralytics**: Fornisce supporto per i modelli YOLO di ultima generazione.
- **Pillow**: Utilizzata per l'elaborazione di immagini (modifica, apertura, salvataggio).
- **opencv-python**: Utilizzata per l'elaborazione di immagini e video.
- **numpy**: Libreria per l'elaborazione numerica e il calcolo matematico.

## Struttura del Progetto
La struttura del progetto è organizzata come segue:

```
.
├── convertions/             # Moduli per la gestione delle conversioni
│   └── mask_convertion.py   # Script per la conversione delle maschere
├── documentation/           # File relativi alla documentazione
│   └── doc.html             # Documentazione HTML del progetto
├── models/                  # Directory per i modelli
│   └── yolo.pt              # Modello YOLO pre-addestrato
├── README.md                # Documentazione principale del progetto
├── configmodel.yaml         # File di configurazione del modello
├── detector.py              # Script per il rilevamento
├── requirements.txt         # Elenco delle dipendenze
├── seg_model.py             # Script per il modello di segmentazione
```

### Descrizione dei File Principali

- **convertions/mask_convertion.py**: Contiene funzioni per la conversione e manipolazione delle maschere.
- **documentation/doc.html**: File HTML per la documentazione dettagliata del progetto.
- **models/yolo.pt**: Modello YOLO pre-addestrato per il rilevamento degli oggetti.
- **configmodel.yaml**: File di configurazione per specificare i parametri del modello.
- **detector.py**: Script principale per il rilevamento degli oggetti.
- **seg_model.py**: Script per la segmentazione delle immagini.

## Come Eseguire
1. Clona o scarica il repository del progetto:

   ```bash
   git clone <url-del-repository>
   cd <nome-cartella>
   ```

2. Installa le dipendenze:

   ```bash
   pip install -r requirements.txt
   ```

3. Avvia lo script principale:

   ```bash
   python detector.py
   ```

## Note
- Assicurati che il tuo ambiente supporti GPU (opzionale ma raccomandato per l'uso con modelli YOLO).
- Verifica di avere i permessi per accedere ai file nella directory `data/` e scrivere in `outputs/`.

## Supporto
Se riscontri problemi, contatta il manutentore del progetto o apri una segnalazione nel repository GitHub.
