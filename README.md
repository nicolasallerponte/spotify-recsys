# Práctica Sistemas de Recomendación — Iteración 1 (KNN Colaborativo)

Este directorio contiene el código y los resultados de la **Iteración 1**, desarrollada para la asignatura de Sistemas de Recomendación del Grado en Ciencia e Ingeniería de Datos (UDC).

## Miembros del Equipo

- **Jacobo Cousillas Taboada** (`jacobo.cousillas@udc.es`)
- **Xaime Paz Ollero** (`xaime.paz.ollero@udc.es`)
- **Nicolas Aller Ponte** (`nicolas.aller@udc.es`)

---

## Contexto y Objetivo

En la Iteración 0 implementamos una propuesta basada en popularidad global que obtuvo un NDCG de **0.0904**. El objetivo de esta iteración es superar ese resultado mediante un enfoque personalizado basado en vecindad.

---

## Filtrado Colaborativo basado en Vecindad (KNN - K-Nearest Neighbors)

El filtrado colaborativo por vecindad estima el score de un track `i` para una playlist `u` a partir de las interacciones de entidades similares. Implementamos las dos variantes definidas en el enunciado:

### User-based

La idea es que busquemos las playlists del dataset de entrenamiento más parecidas a la tuya y recomendemos lo que escuchan. Para una playlist de test con unas semillas dadas, buscamos las `k` playlists de train con mayor similitud coseno. Luego agregamos sus tracks ponderados por esa similitud — los tracks que aparecen en muchas playlists similares y con alta similitud suben en el ranking.

Es similar a una recomendación del tipo "usuarios con gustos parecidos también escucharon esto".

### Item-based

La idea es que para cada track de la semilla, busquemos los tracks más similares a él y los recomendemos. Dos tracks son similares si aparecen juntos en las mismas playlists. Para cada track semilla `j` encontramos sus `k` vecinos más similares y acumulamos esas similitudes en sus scores — un track candidato que es similar a varios tracks de la semilla acumula más puntuación.

Es similar a una recomendación del tipo "porque escuchas X, Y, Z, aquí hay canciones parecidas".

### Similitud coseno

En ambos casos utilizamos similitud coseno, tanto para calcular el vecindario como para ponderar las contribuciones al score. La matriz de interacciones es binaria (1 si el track está en la playlist, 0 si no), por lo que la similitud coseno entre dos entidades mide la proporción de elementos compartidos normalizada por el tamaño de cada una.

---

## Decisiones de Implementación

### Representación sparse

El dataset tiene ~1M playlists y ~2M tracks únicos. La matriz de interacciones es extremadamente dispersa (~0.006% de densidad), lo que hace inviable cualquier representación densa. Utilizamos matrices CSR (`csr_matrix`) para operaciones por filas y CSC (`csc_matrix`) para acceso eficiente por columnas.

### Acceso por columnas CSC en lugar de producto matricial denso

El cuello de botella inicial en user-based era el producto `seed_vector @ M.T`, que internamente genera un vector denso de 1M elementos para cada playlist de test. Lo sustituimos por `M_csc[:, seed_indices].sum(axis=1)`, que extrae únicamente las columnas correspondientes a los tracks semilla operando solo sobre los elementos no nulos de esas columnas, de esta forma reducimos el coste porque solo tocamos las playlists que comparten al menos un track con la semilla.

### Item-based: formulación invertida por semilla

La formulación directa de item-based requeriría calcular `sim(i, j)` para todos los 2M tracks `i` y cada track semilla `j`, produciendo matrices de tamaño inmanejable. En cambio, invertimos el punto de vista: para cada track semilla `j`, calculamos sus `k` vecinos más similares y acumulamos `sim(vecino, j)` en los scores de esos vecinos. Esto es equivalente a la formula que se nos da originalmente pero opera con vectores de dimensión `n_playlists` (~1M) en vez de `n_tracks` (~2M), y además aprovecha la sparsidad del vector de cada track.

### Fallback a popularidad

Las playlists con semilla vacía (0 tracks conocidos) no tienen información para calcular similitudes, por lo que devuelven 0 vecinos. Para estas y para cualquier playlist que no alcance las 500 recomendaciones por KNN, completamos con el ranking de popularidad global de la Iteración 0. En la configuración final con k=500, **1015 de las 10000 playlists** necesitaron este fallback, correspondiendo principalmente a las playlists de la categoría de 0 semillas.

### Paralelización (descartada)

Exploramos paralelizar el bucle de playlists con `joblib.Parallel` usando threads, ya que scipy/numpy liberan el GIL en operaciones matriciales. Sin embargo, con 8 cores y tras intentarlo de diversas maneras, no conseguimos una buena implementación, más adelante en próximas iteraciones se volverá a valorar. Descartamos esta vía momentáneamente y mantuvimos el procesamiento secuencial, que con las optimizaciones de acceso sparse resulta suficientemente eficiente.

---

## Estructura del Código

- `src/data_loader.py`: Procesa el ZIP de entrenamiento y construye la matriz CSR de interacciones, los diccionarios de mapeo y el ranking de popularidad. Genera `data/processed/`.
- `src/baseline.py`: Baseline de popularidad global (Iteración 0). Genera `submissions/iteracion_0_baseline.csv`.
- `src/knn.py`: Implementación principal de KNN colaborativo. Acepta `--mode user/item` y `--k`. Genera el CSV de submission correspondiente.
- `src/evaluation.py`: Calcula R-Precision, NDCG y Clicks comparando una submission con el ground truth. Acepta el nombre del fichero como argumento.
- `src/verify_submission.py`: Verifica que el CSV cumple el formato del reto (500 tracks, sin duplicados, URIs válidas).

---

## Instrucciones de Ejecución

```bash
# Instalar dependencias
uv sync
source .venv/bin/activate

# (Solo si no existe data/processed/) Procesar dataset de entrenamiento
python src/data_loader.py

# Generar recomendaciones
python src/knn.py --mode user --k 500
python src/knn.py --mode item --k 500

# Evaluar
python src/evaluation.py iteracion_1_knn_user_k500.csv
python src/evaluation.py iteracion_1_knn_item_k500.csv

# Verificar formato
python src/verify_submission.py iteracion_1_knn_user_k500.csv
```

---

## Resultados

### Comparativa completa

| Método         | k       | R-Precision  | NDCG         | Clicks     |
| -------------- | ------- | ------------ | ------------ | ---------- |
| Baseline (it0) | —       | 0.025670     | 0.090437     | 17.3094    |
| User-based     | 100     | 0.157843     | 0.321931     | 4.8453     |
| User-based     | 250     | 0.159274     | 0.339983     | 4.5208     |
| User-based     | **500** | **0.158728** | **0.344748** | **4.4764** |
| Item-based     | 100     | 0.148397     | 0.305787     | 5.9639     |
| Item-based     | 250     | 0.151660     | 0.322691     | 5.8864     |
| Item-based     | 500     | 0.153670     | 0.329705     | 5.8912     |

### Análisis

**Mejora respecto al baseline**: ambas variantes de KNN suponen una mejora muy significativa sobre el baseline de popularidad. El NDCG pasa de 0.090 a ~0.345 en la mejor configuración — casi **4x de mejora**. Esto confirma que la personalización por vecindad aporta mucho valor frente a un ranking global.

**User-based vs Item-based**: user-based obtiene mejores resultados en todas las métricas y es además más rápido (~2 minutos frente a ~40 minutos para k=500). La ventaja de user-based en este dataset tiene sentido ya que las playlists del MPD son entidades ricas con decenas o cientos de tracks, lo que hace que la similitud entre playlists sea una señal muy informativa. En cambio, la similitud entre tracks depende de cuántas playlists comparten, y muchos tracks tienen vectores muy dispersos que producen similitudes poco fiables.

**Efecto de k en user-based**: aumentar k de 100 a 500 mejora el NDCG de 0.322 a 0.345. La mejora se va reduciendo, de 100 a 250 hay una ganancia mayor que de 250 a 500, lo que sugiere que el óptimo está en torno a k=500 o algo superior, pero con rendimientos decrecientes.

**Mejor configuración**: `--mode user --k 500` con NDCG=**0.344748**.
