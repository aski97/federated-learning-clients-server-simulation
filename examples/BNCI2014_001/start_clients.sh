#!/bin/bash

# Crea una cartella per i log, se non esiste
mkdir -p client_logs

echo "Avvio di 10 client in background (con log in /client_logs)..."

for ((i=1; i<=10; i++)); do
    # Avvia il client e salva il suo output (stdout e stderr)
    # in un file di log separato.
    python Client.py $i > client_logs/client_$i.log 2>&1 &
done

echo "Tutti i client sono stati avviati."