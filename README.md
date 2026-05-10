# Práctica Sistemas de Recomendación - Iteraciones 0, 1, 2 y 3

Este directorio contiene el código y los resultados de las **Iteraciones 1 y 2**, desarrolladas para la asignatura de Sistemas de Recomendación del Grado en Ciencia e Ingeniería de Datos (UDC).

## Miembros del Equipo

- **Jacobo Cousillas Taboada** (`jacobo.cousillas@udc.es`)
- **Xaime Paz Ollero** (`xaime.paz.ollero@udc.es`)
- **Nicolas Aller Ponte** (`nicolas.aller@udc.es`)

---

## Contexto y Objetivo

Este repositorio documenta cuatro iteraciones progresivas de sistemas de recomendación aplicados al Spotify Million Playlist Dataset Challenge. En cada iteración se aumenta la sofisticación del método:

- **Iteración 0**: Baseline de popularidad global → NDCG **0.090**
- **Iteración 1**: Filtrado colaborativo por vecindad (KNN user/item-based) → NDCG **0.345**
- **Iteración 2**: Factorización matricial PureSVD (inductive y transductive) → NDCG **0.288**
- **Iteración 3**: Métodos lineales escasos SLIM y FISM con dataset reducido → NDCG **0.339**

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
- `src/puresvd.py`: Implementación de PureSVD (Iteración 2). Acepta `--mode inductive|transductive` y `--k`.
- `src/slim.py`: Implementación de SLIM y FISM (Iteración 3). Lee de `data/trimmed_dataset.zip`. Acepta `--mode slim|fism` y múltiples hiperparámetros.
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

---

## Iteración 3 - SLIM & FISM

### Algoritmos

**SLIM** aprende una matriz de similitud ítem-ítem W resolviendo:

```
min  ||X – XW||²_F  +  λ||W||₁  +  β||W||²_F   s.t.  diag(W) = 0,  W ≥ 0
```

**FISM** factoriza esa matriz como S = PQᵀ con P, Q ∈ ℝ^{n×f}, usando la misma pérdida con regularización L1+L2 sobre P y Q. El scoring es `ŷ(u,:) = (x_u @ P) @ Qᵀ`, sin materializar PQᵀ.

---

### Dataset

SLIM requiere una matriz W de n_items × n_items. Con el dataset completo (~2.26M tracks) eso son ~20 PB - inviable. El dataset trimado (`data/trimmed_dataset.zip`, 1704 tracks) hace W manejable (~11.6 MB).

| Conjunto | Playlists | Tracks | Interacciones | Densidad |
|----------|-----------|--------|---------------|----------|
| Train    | 4 347     | 1 704  | 6 779         | 0.092%   |
| Test     | 29        | -      | -             | -        |

20 de 29 playlists de test tienen 0 semillas (cold-start por diseño del dataset).

---

### Decisiones de implementación

**`reduce_sum` en lugar de `reduce_mean`:** con `reduce_mean` el gradiente de reconstrucción por co-ocurrencia queda en ~2.7×10⁻⁷ (dividido entre 7.4M elementos), aplastado por el gradiente de L1 incluso con λ=0.001. W converge a cero. Con `reduce_sum` el gradiente es ~2 por co-ocurrencia - del mismo orden que la regularización - y W aprende correctamente (99.8% sparse, capturando las ~5400 co-ocurrencias reales).

**Adam en lugar de SGD:** el esqueleto usa SGD, pero con gradientes de escala muy variable (datos ultra-sparse) SGD requiere ajuste fino del learning rate. Adam adapta la tasa por parámetro y converge en ~500 épocas sin tunear.

**Proyección post-gradiente (SLIM):** las restricciones `diag(W)=0` y `W≥0` se aplican como proyección tras cada actualización (`W.assign(tf.maximum(W, 0.0) * diag_mask)`). `diag(W)=0` evita la solución trivial W=I; `W≥0` hace los scores interpretables como similitudes.

**Fallback a popularidad:** playlists sin semillas (o con menos de 500 candidatos con score>0) se completan con el ranking de popularidad del training. Para FISM, sin restricción W≥0, los scores pueden ser negativos → más dependencia del fallback.

---

### Ejecución

```bash
python src/slim.py --mode slim --epochs 500 --lr 0.01 --lambda-a 5.0 --lambda-b 1.0
python src/slim.py --mode fism --epochs 2000 --lr 0.001 --lambda-a 1.0 --lambda-b 0.0 --factors 64
```

---

### Resultados

| Método                      | Epochs | f  | R-Precision  | NDCG         | Clicks      |
| --------------------------- | ------ | -- | ------------ | ------------ | ----------- |
| Baseline (it0)              | -      | -  | 0.025670     | 0.090437     | 17.3094     |
| KNN User-based (it1)        | -      | 500| 0.158728     | 0.344748     | 4.4764      |
| PureSVD inductive (it2)     | -      | 100| 0.127230     | 0.287612     | 5.7683      |
| **SLIM** (λ=5, β=1)        | 500    | -  | 0.068966     | **0.339358** | **5.7586**  |
| FISM (f=64, λ=1, β=0)      | 2000   | 64 | 0.103448     | 0.291069     | 11.7931     |

*Evaluación sobre los 29 playlists del dataset trimado - no comparable directamente con it1/it2 (datasets distintos).*

SLIM iguala prácticamente al KNN user-based (NDCG 0.339 vs 0.345) aprendiendo las co-ocurrencias mediante optimización en lugar de calcularlas directamente. FISM pierde precisión porque la factorización de rango bajo (f=64) no representa bien los ~5400 pares con co-ocurrencia real en un dataset tan sparse.
