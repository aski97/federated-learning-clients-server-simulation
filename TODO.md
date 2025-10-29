# TODO — attività immediate (esteso)

Questa versione estesa raccoglie bugfix, refactor e funzionalità suggerite per rendere il framework pronto per esperimenti riproducibili, produzione leggera e sviluppo continuo.

## Priorità Critica (fix / stabilità)
1. Aggregation / weights format
   - Uniformare rappresentazione: usare list[numpy.ndarray] (layer-wise) ovunque.
   - Aggiornare AggregationAlgorithm e tutte le implementazioni (FedAvg, FedAdam, FedAvgMomentum, FedSGD, FedMiddleAvg).
   - Aggiungere test che verificano compatibilità con keras.Model.get/set_weights().

2. Networking / socket lifecycle
   - Garantire che TCPServer.run blocchi correttamente fino a shutdown; fix già applicato.
   - Rendere accept timeout configurabile (None = blocking).
   - Gestire reconnection e retry lato client (parametrizzabili).
   - Aggiungere heartbeat/keepalive e gestione dei client disconnect parziali.

3. Messaggistica / serializzazione
   - Documentare e validare schema messaggi (versioning).
   - Valutare sostituzione pickle con msgpack/flatbuffers/protobuf per performance e sicurezza.
   - Limitare dimensione massima messaggi e gestire streaming chunked per payload grandi.

4. Logging
   - Rimuovere logging.basicConfig dai moduli; configurazione centralizzata negli esempi.
   - Adottare logger namespaced e supportare formato strutturato (JSON) opzionale.
   - Aggiungere livelli DEBUG/TRACE per diagnostica e metriche runtime.

## Funzionalità e miglioramenti (features)
1. Profiling client/server (migliorare)
   - Persistenza per-client dei risultati di profiling (file o endpoint).
   - Modalità a basso-overhead: campionamento temporale e statistico (es. solo percentili).
   - Profiling selettivo per round/epoca, disable automatico per grandi sperimenti.
   - API per esportare risultati (CSV/JSON/npz).

2. Fault tolerance & resuming
   - Checkpointing federated model e round metadata.
   - Ripresa automatica da checkpoint.
   - Supporto partial participation (client mancanti in un round) e timeout per round.

3. Scalabilità e orchestrazione
   - Supporto asincrono (opzionale) per Federated Averaging con client che arrivano in tempi diversi.
   - Docker/Docker Compose esempi per avviare server + N client.
   - Kubernetes manifest + Helm chart (per esperimenti su cluster).

4. Privacy & sicurezza
   - Canale TLS opzionale (wrap socket con ssl).
   - Supporto per secure aggregation (opzionale) e/ o differential privacy (noise injection lato client).
   - Autenticazione semplice (token) ed autorizzazioni minime.

5. Compressione e efficienza
   - Compressione pesi/gradients (quantization, subsampling, gzip, msgpack).
   - Trasferimento incrementale (send only diffs) per ridurre banda.

6. Client selection & sampling
   - Strategie di selezione clients (random, round-robin, weighted).
   - Configurare percentuale di partecipazione per round.

## Refactor / Architettura
1. Separare componenti
   - networking (TCPServer/Client), aggregation, utils, profiling, io.
   - Plugin system per algoritmi di aggregazione e metriche.

2. API e configurazione
   - File di configurazione YAML/JSON per esperimenti (server, clients, dataset, profiling).
   - CLI centralizzata per lanciare server/clients con args.

3. Testability
   - Mock network layer per test unitari (senza socket reali).
   - Coverage minima per aggregation e CSUtils.

4. Type-safety & lint
   - Aggiungere type hints coerenti, mypy e rimozione uso di dtype=object.
   - Configurare lint (ruff/flake8) e formatter (black).

## Testing & CI
1. Unit tests:
   - AggregationAlgorithm (weighted/unweighted), FedAdam/FedAvgMomentum per-layer.
   - CSUtils serialize/unpack.
   - TCPServer/TCPClient integration test minimal (loopback).
2. Integration tests:
   - E2E su macchina locale con N client finti (fast, small model).
3. CI:
   - GitHub Actions pipeline (lint, mypy, pytest).
   - Artefatto test-report e opzione per run di smoke tests su push.

## UX / Documentazione
1. README e HOWTO aggiornati (esempi: avviare server, avviare N client, profiling).
2. Notebook Jupyter con esperimenti di esempio e grafici.
3. Documentazione API per sviluppatori (docstrings e un README per ogni modulo).

## Monitoring e metriche
1. Esportare metriche (Prometheus) su server: round, clients_connected, bandwidth, latency.
2. Dashboard minimale (Grafana) come esempio.

## Performance / benchmark
1. Script benchmark per misurare:
   - throughput di rete per round,
   - tempo di aggregazione su modelli di diverse dimensioni,
   - memoria peak lato client/server.
2. Esempi di modelli più grandi per testare scalabilità e compressione.

## Esempi avanzati / ricerca
1. Supporto per heterogenous clients (differenti modelli o feature sets).
2. Algoritmi più avanzati: FedProx, SCAFFOLD, secure aggregation, federated transfer learning.
3. Modularità per plug-in di ottimizzatori lato server (Adam, Momentum, etc).

---

## Azioni consigliate immediate (ordine operativo)
1. Rifattorizzare AggregationAlgorithm e tutti gli algoritmi per operare layer-wise.
2. Centralizzare logging e rimuovere basicConfig dai moduli.
3. Implementare e testare retry lato client e accept_timeout configurabile.
4. Sostituire pickle per serializzazione o definire un wrapper sicuro/validato.
5. Aggiungere 10-20 unit tests chiave e pipeline CI minima.
6. Fornire esempi Docker + README aggiornato.