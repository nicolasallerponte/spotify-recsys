# Práctica Sistemas de Recomendación - Iteraciones 1 y 2

Este directorio contiene el código y los resultados de las **Iteraciones 1 y 2**, desarrolladas para la asignatura de Sistemas de Recomendación del Grado en Ciencia e Ingeniería de Datos (UDC).

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

La idea es que busquemos las playlists del dataset de entrenamiento más parecidas a la tuya y recomendemos lo que escuchan. Para una playlist de test con unas semillas dadas, buscamos las `k` playlists de train con mayor similitud coseno. Luego agregamos sus tracks ponderados por esa similitud - los tracks que aparecen en muchas playlists similares y con alta similitud suben en el ranking.

Es similar a una recomendación del tipo "usuarios con gustos parecidos también escucharon esto".

### Item-based

La idea es que para cada track de la semilla, busquemos los tracks más similares a él y los recomendemos. Dos tracks son similares si aparecen juntos en las mismas playlists. Para cada track semilla `j` encontramos sus `k` vecinos más similares y acumulamos esas similitudes en sus scores - un track candidato que es similar a varios tracks de la semilla acumula más puntuación.

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

## Resultados - Iteración 1

### Comparativa completa

| Método         | k       | R-Precision  | NDCG         | Clicks     |
| -------------- | ------- | ------------ | ------------ | ---------- |
| Baseline (it0) | -       | 0.025670     | 0.090437     | 17.3094    |
| User-based     | 100     | 0.157843     | 0.321931     | 4.8453     |
| User-based     | 250     | 0.159274     | 0.339983     | 4.5208     |
| User-based     | **500** | **0.158728** | **0.344748** | **4.4764** |
| Item-based     | 100     | 0.148397     | 0.305787     | 5.9639     |
| Item-based     | 250     | 0.151660     | 0.322691     | 5.8864     |
| Item-based     | 500     | 0.153670     | 0.329705     | 5.8912     |

### Análisis

**Mejora respecto al baseline**: ambas variantes de KNN suponen una mejora muy significativa sobre el baseline de popularidad. El NDCG pasa de 0.090 a ~0.345 en la mejor configuración - casi **4x de mejora**. Esto confirma que la personalización por vecindad aporta mucho valor frente a un ranking global.

**User-based vs Item-based**: user-based obtiene mejores resultados en todas las métricas y es además más rápido (~2 minutos frente a ~40 minutos para k=500). La ventaja de user-based en este dataset tiene sentido ya que las playlists del MPD son entidades ricas con decenas o cientos de tracks, lo que hace que la similitud entre playlists sea una señal muy informativa. En cambio, la similitud entre tracks depende de cuántas playlists comparten, y muchos tracks tienen vectores muy dispersos que producen similitudes poco fiables.

**Efecto de k en user-based**: aumentar k de 100 a 500 mejora el NDCG de 0.322 a 0.345. La mejora se va reduciendo, de 100 a 250 hay una ganancia mayor que de 250 a 500, lo que sugiere que el óptimo está en torno a k=500 o algo superior, pero con rendimientos decrecientes.

**Mejor configuración**: `--mode user --k 500` con NDCG=**0.344748**.

---

## Iteración 2 - PureSVD (Factorización Matricial)

### Algoritmo: PureSVD

En vez de calcular similitudes explícitas entre playlists o tracks, PureSVD factoriza la matriz de interacciones en un espacio de dimensión reducida (factores latentes) mediante SVD truncada:

```
R ≈ Ũ × Σ̃ × Ṽᵀ
```

donde `k` (número de factores latentes) es el hiperparámetro principal. Las componentes capturan patrones de co-ocurrencia implícitos entre playlists y tracks.

### Variante Inductive (solo datos de entrenamiento)

La SVD se calcula únicamente sobre las 990.000 playlists de entrenamiento. Para cada playlist de test, se proyecta su vector de semillas al espacio latente y se puntúan todos los tracks:

```
ū_new = r̄_new × Ṽ × Σ⁻¹
ĝ_new,i = ū_new × Σ × v̄ᵢᵀ
```

Algebraicamente, Σ se cancela: `ĝ = (r̄_new × Ṽ) × Ṽᵀ`, lo que equivale a sumar las filas de Ṽ correspondientes a los tracks semilla y proyectar sobre Ṽᵀ. Esta implementación es eficiente porque solo accede a las columnas de la semilla.

### Variante Transductive (train + test juntos)

La SVD se calcula sobre la matriz combinada (990.000 train + 10.000 test). Las filas del test ya están en el espacio latente directamente:

```
ĝ_u,i = ū_u × Σ × v̄ᵢᵀ
```

Teóricamente más precisa porque la descomposición ve las semillas de test durante el entrenamiento. En la práctica el tiempo de SVD aumenta ligeramente.

### Decisiones de Implementación

#### Orden de valores singulares

`scipy.sparse.linalg.svds` devuelve los valores singulares en **orden ascendente** (el más pequeño primero). Se invierten inmediatamente tras la llamada para que los factores más importantes queden primero, aunque matemáticamente el orden no afecta al scoring.

#### Gestión de memoria

La matriz Ṽᵀ ocupa `k × 2.26M × 4 bytes` (float32): ~0.45 GB para k=50, ~1.8 GB para k=200. Para minimizar el uso de RAM:
- Se castea a float32 justo al salir de `svds` (ahorra ~50% respecto a float64).
- En modo inductive, la matriz U (990k × k) se elimina tras la SVD ya que no se usa para scoring.
- En modo transductive, se extrae solo el bloque de test (`U[n_train:]`) y se elimina el resto.

#### Playlists sin semilla

Las 1.000 playlists de test con 0 tracks conocidos producen un vector de scores nulo → caída completa al fallback de popularidad global.

---

## Estructura del Código (Iteración 2)

- `src/puresvd.py`: Implementación de PureSVD. Acepta `--mode inductive|transductive` y `--k`. Genera el CSV de submission correspondiente.

---

## Instrucciones de Ejecución (Iteración 2)

```bash
# Generar recomendaciones PureSVD
python src/puresvd.py --mode inductive --k 50
python src/puresvd.py --mode inductive --k 100
python src/puresvd.py --mode inductive --k 200
python src/puresvd.py --mode transductive --k 50
python src/puresvd.py --mode transductive --k 100
python src/puresvd.py --mode transductive --k 200

# Evaluar
python src/evaluation.py iteracion_2_puresvd_inductive_k100.csv

# Verificar formato
python src/verify_submission.py iteracion_2_puresvd_inductive_k100.csv
```

---

## Resultados - Iteración 2

### Comparativa completa

| Método                   | f       | R-Precision  | NDCG         | Clicks      |
| ------------------------ | ------- | ------------ | ------------ | ----------- |
| Baseline (it0)           | -       | 0.025670     | 0.090437     | 17.3094     |
| KNN User-based (it1)     | 500     | 0.158728     | 0.344748     | 4.4764      |
| PureSVD inductive        | 50      | 0.119531     | 0.277740     | 6.4752      |
| PureSVD inductive        | **100** | **0.127230** | **0.287612** | **5.7683**  |
| PureSVD inductive        | 200     | 0.129344     | 0.287225     | 5.3967      |
| PureSVD transductive     | 50      | 0.119502     | 0.277722     | 6.4745      |
| PureSVD transductive     | **100** | **0.127219** | **0.287587** | **5.7706**  |
| PureSVD transductive     | 200     | 0.129283     | 0.287129     | 5.3999      |

### Análisis

**PureSVD vs KNN**: PureSVD no supera a KNN user-based en este dataset (NDCG ~0.288 frente a 0.345). Esto era esperable ya que la matriz es extremadamente sparse (0.003% de densidad) y KNN user-based aprovecha directamente las co-ocurrencias exactas entre playlists, mientras que SVD con pocos factores latentes no captura suficiente varianza para superar esa señal directa.

**Efecto de f**: el salto de f=50 a f=100 aporta una mejora apreciable en NDCG (+0.01) y Clicks (-0.7). De f=100 a f=200 la ganancia en NDCG se estanca o incluso retrocede ligeramente, lo que sugiere que el óptimo está cerca de f=100 para este rango. Aumentar f (k en nuestro codigo) mejora R-Precision (más tracks relevantes encontrados) pero no mejora el ranking.

**Inductive vs Transductive**: los resultados son prácticamente idénticos (diferencias en el cuarto decimal). Esto indica que las semillas de test son tan escasas respecto al volumen de entrenamiento (990k playlists) que incluirlas en la factorización apenas altera los vectores latentes. La variante inductive es por tanto la opción práctica: mismo rendimiento, sin necesidad de refactorizar cuando llegan nuevas playlists.

**Mejor configuración**: `--mode inductive --f 100` con NDCG=**0.287612** (o transductive con resultado casi idéntico).
