# TODO — attività immediate (aggiornato)

Aggiornamento esteso del TODO con focus su profiling client-side e priorità operative. Elenco sintetico e azioni concrete per le prossime modifiche.

## Obiettivi principali
- Stabilità e coerenza dei formati (weights/gradients come list[numpy.ndarray]).
- Robustezza rete (shutdown, retry, timeouts configurabili).
- Profiling client efficiente, persistente e configurabile.
- Serializzazione sicura/performante e test coverage minimo.

---

## Priorità Critica (da fare subito)
1. Aggregation / weights format
   - Rifattorizzare AggregationAlgorithm (già iniziato) e adattare tutte le chiamate nel codice per usare list[numpy.ndarray].
   - Aggiungere assert/validator che controllino che self.weights e client payload siano liste prima di aggregare.
   - Test unitari semplici che verificano compatibilità con keras.Model.get_weights()/set_weights().

2. Networking / lifecycle
   - Assicurare che `TCPServer.run()` blocchi sullo _stop_event e non chiuda la socket prematuramente.
   - Rendere accept timeout e comportamenti di shutdown configurabili via argomento/attributo.
   - Implementare retry esponenziale lato client con parametri configurabili (_connect_retries, _connect_retry_delay).

3. Messaggistica / serializzazione
   - Definire e documentare schema dei messaggi (versioning minimale).
   - Estrarre/implementare `serialize_message` / `deserialize_message` in `CSUtils.py` usando un wrapper che può essere cambiato (pickle per ora, possibilità di passare a msgpack/protobuf).
   - Limitare una dimensione massima opzionale per i messaggi e gestire chunking se necessario.

4. Logging
   - Rimuovere `logging.basicConfig` dai moduli della libreria; lasciare configurazione nel top-level (es. examples/Server.py).
   - Usare nomi logger coerenti (`federated_sim.server`, `federated_sim.client`, `federated_sim.agg`).

---

## Profiling client (design e tasks)
Obiettivo: profiling utile per debug/benchmark ma non intrusivo per esperimenti su larga scala.

1. Design delle modalità (implementate/da finalizzare)
   - none: disabilitato.
   - light: tempo totale di training + peak RAM via resource (basso overhead).
   - tracemalloc: memoria dettagliata via tracemalloc (moderato).
   - trace: trace delle istruzioni (alto overhead) — usare solo quando necessario.
   - sampled: modalità che campiona N epoche o 1 epoca ogni K per ridurre overhead.

2. Tasks pratici
   - Centralizzare modalità e parametri di profiling in `TCPClient` (`_profiling_mode`, `_profiling_sample_per_epoch`, `_profiling_sample_rate`).
   - Attivazione via messaggio server (m_body['configurations']): profiling (bool), profiling_mode (str), sample_per_epoch (bool), sample_rate (int).
   - Persistenza: salvare risultati profiling per client in file JSON/npz nella directory `logs/profiling/client_{id}_round_{r}.json` (opzionale) — implementare writer asincrono o a fine round.
   - Formato di output standard: { round, mode, total_time, peak_mem, epoch_times:[], epoch_mem_peaks:[], instructions (if trace) }.
   - API per esportare i risultati aggregati sul server (ricevere e salvare `info_profiling` nel CLIENT_EVALUATION): server salva per-client file e opzionalmente genera CSV/JSON aggregato.

3. Ottimizzazioni
   - Evitare `trace` di default; usarlo solo su richiesta e loggare un avviso.
   - Usare tracemalloc per memorie dettagliate; fallback a resource su piattaforme non supportate.
   - Per grandi esperimenti, raccomandare `light + sampled` come default.

---

## Priorità Media (migliorie e refactor)
1. Persistenza / Resume
   - Checkpoint federated model (.npz/.npy) e metadata (round corrente).
   - API `save_federated_weights` / `load_initial_weights` robuste (gestire permissive formats).

2. Tests
   - Unit tests: aggregation, CSUtils serialize/unpack, client connect retry, server accept loop (mock).
   - Integration test minimal server+1 client (loopback).

3. CLI / config
   - Aggiungere file di configurazione YAML per esperimenti (server address, n_clients, profiling_mode, timeouts).
   - CLI wrapper per lanciare server e client con config.

4. Security & performance
   - Valutare switch a msgpack/protobuf per messaggi pesanti.
   - Aggiungere opzione TLS (ssl.wrap_socket) per canale sicuro.

---

## Priorità Bassa (feature, deployment)
- Docker / docker-compose per avviare server + N clients.
- Endpoint metrics Prometheus + esempio dashboard.
- Compressione/quantizzazione dei pesi per uso in rete limitata.
- Algoritmi avanzati: FedProx, SCAFFOLD, secure aggregation.

---

## Esempio di next-steps immediati (ordine operativo)
1. Finalizzare `CSUtils.serialize_message` / `deserialize_message` e testare roundtrip.
2. Applicare i fix minimi suggeriti per TCPServer/TCPClient (weights come list, retry client, blocking run).
3. Implementare salvataggio profiling lato client (JSON) e invio `info_profiling` al server.
4. Aggiungere 4-6 unit tests per aggregation e message serialization.
5. Rimuovere basicConfig dai moduli e aggiungere logging config in `examples/BNCI2014_001/Server.py`.