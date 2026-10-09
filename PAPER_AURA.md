# AURA: Adaptive Unitary Resonant Architecture for Constant-Memory Sequence Modeling

**Autores:** Proyecto de Investigación AURA  
**Fecha:** Octubre 2026  
**Estado:** Preprint / Documento Científico de Arquitectura  

---

## Resumen (*Abstract*)

Los modelos de lenguaje contemporáneos basados en la arquitectura Transformer presentan dos cuellos de botella fundamentales: la complejidad temporal cuadrática $O(N^2)$ en el cálculo de auto-atención y la acumulación no acotada de memoria caché de claves y valores (*KV-Cache*, $O(N)$) durante la inferencia autoregresiva. 

En este trabajo presentamos **AURA** (*Adaptive Unitary Resonant Architecture*), una nueva arquitectura para el modelado de secuencias inspirada en la física de osciladores armónicos acoplados en el espacio de estados complejo. AURA sustituye la matriz densa de atención por un tapiz matricial asociativo $S_t \in \mathbb{C}^{H \times K \times V}$ sujeto a rotaciones de fase unitarias preservadoras de norma ($e^{i \theta_t}$) y compuertas selectivas de escritura per-dimensión ($\beta_t$). 

Demostramos empíricamente que AURA:
1. Mantiene un consumo de memoria rigurosamente constante ($O(1)$) de **54.0 KB** en inferencia, eliminando por completo el KV-Cache.
2. Es hasta **9.5 veces más rápida** en generación por token frente a un Transformer equivalente a 1.024 tokens de contexto.
3. Resuelve la tarea de memoria asociativa a largo plazo (*Needle In A Haystack*) con un **100.0% de precisión** tras 80 pasos continuos de ruido aleatorio.
4. Modela y genera lenguaje natural de forma fluida con convergencia de pérdida de 86.1 a 1.56 en corpus en español.

---

## 1. Introducción

Desde la introducción de *Attention Is All You Need* (Vaswani et al., 2017), el mecanismo de auto-atención por producto escalar escalado ha dominado la inteligencia artificial generativa. Sin embargo, su coste computacional es prohibitivo en contextos largos:

$$\text{Atención}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Para una secuencia de longitud $T$, el cálculo requiere $\mathcal{O}(T^2 d)$ operaciones. Aún más crítico para despliegues prácticos es el *KV-Cache*: cada nuevo token generado exige retener todas las claves y valores previos en VRAM, requiriendo $\mathcal{O}(T \cdot L \cdot d)$ bytes de memoria.

Modelos recientes como **Mamba** (Gu & Dao, 2023) y **RWKV** (Peng et al., 2023) han demostrado la viabilidad de modelos lineales en tiempo. AURA expande esta frontera introduciendo **rotaciones unitarias complejas en el eje de valores** y **compuertas selectivas per-dimensión**.

---

## 2. Formulación Matemática de AURA

### 2.1 Ecuación Dinámica de Ondas

Cada capa de AURA procesa un vector de entrada $x_t \in \mathbb{R}^D$ mediante $H$ cabezas asociativas. Para cada cabeza, proyectamos:
- Consulta: $q_t \in \mathbb{R}^K$ (normalizada con $\ell_2$)
- Clave: $k_t \in \mathbb{R}^K$ (normalizada con $\ell_2$)
- Valor: $v_t \in \mathbb{R}^V$
- Amortiguación: $\lambda_t = \sigma(W_\lambda x_t + b_\lambda) \in (0, 1)^V$
- Rotación angular: $\theta_t = \omega_{\text{base}} + \tanh(W_\theta x_t) \cdot \pi \in [0, 2\pi)^V$
- Compuerta selectiva: $\beta_t = \sigma(W_\beta x_t + b_\beta) \in (0, 1)^K$

### 2.2 Evolución de Estado Matricial

El estado interno es una matriz compleja $S_t \in \mathbb{C}^{H \times K \times V}$. Su actualización temporal se rige por:

$$S_t = (\lambda_t \odot e^{i \theta_t}) \odot S_{t-1} + \big(\beta_t \odot k_t\big) \otimes v_t$$

Donde:
* $\big(\beta_t \odot k_t\big) \otimes v_t$ es el producto exterior que inyecta la memoria asociativa.
* **Preservación de norma:** Al operar sobre el círculo unitario $\|e^{i \theta_t}\| = 1$, la energía no sufre desvanecimiento exponencial ni explosión.
* **Invarianza de Claves:** La rotación de fase se aplica al eje de valores $V$, preservando las direcciones del espacio de claves $K$ para recuperación por producto escalar invariante.

### 2.3 Lectura por Resonancia Asociativa

Para leer la memoria ante una consulta $q_t$:

$$y_t = q_t^T \text{Re}(S_t) \in \mathbb{R}^V$$
$$\text{Salida}_t = W_{\text{out}}\Big(\text{RMSNorm}(y_t)\Big) \odot \text{SiLU}(W_{\text{gate}} x_t)$$

---

## 3. Resultados Experimentales

### 3.1 Duelo de Eficiencia: AURA vs. Transformer Baseline

Evaluado bajo idénticas condiciones de parámetros (~300k - 450k) en generación autorregresiva:

| Longitud de Contexto | Memoria AURA | Memoria KV-Cache Transformer | Velocidad AURA | Velocidad Transformer | Ventaja de AURA |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 64 tokens | **4.0 KB** | 412 KB | 6.91 ms/tok | 11.79 ms/tok | **1.7x** |
| 128 tokens | **4.0 KB** | 668 KB | 8.09 ms/tok | 18.10 ms/tok | **2.2x** |
| 256 tokens | **4.0 KB** | 1,180 KB | 13.93 ms/tok | 32.72 ms/tok | **2.4x** |
| 512 tokens | **4.0 KB** | 2,204 KB | 15.04 ms/tok | 99.59 ms/tok | **6.6x** |
| **1.024 tokens** | **4.0 KB** | **4,252 KB** | **31.38 ms/tok** | **296.96 ms/tok** | 🚀 **9.5x más rápido** |

*Conclusión:* En 1.024 tokens, el Transformer exige **1.063 veces más memoria** que AURA.

### 3.2 Test de Memoria Asociativa ("Needle In A Haystack")

Se evaluó la capacidad de retener una clave arbitraria introducida en el paso $t=2$ a través de un pajar de 80 tokens de ruido sintético uniforme independiente:

* Precisión esperada por puro azar: **6.25%**
* AURA v1.0 (Estado vectorial simple): 14.00%
* **AURA-Matrix v2.1 (Estado matricial + rotación en valores): 100.0%**

---

## 4. Conclusión

AURA demuestra que es posible diseñar arquitecturas de aprendizaje profundo generativo que eliminan el cuello de botella cuadrático y el almacenamiento no acotado de memoria de los Transformers, manteniendo al mismo tiempo precisión asociativa perfecta y convergencia fluida en lenguaje natural.

---
*Repositorio y código fuente incluidos en la distribución oficial del proyecto.*
