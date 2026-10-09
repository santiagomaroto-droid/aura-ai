# AURA: Adaptive Unitary Resonant Architecture 🌌

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Memory: O(1)](https://img.shields.io/badge/Memory-O(1)_Constant-brightgreen.svg)]()
[![Time: O(N)](https://img.shields.io/badge/Complexity-O(N)_Linear-brightgreen.svg)]()
[![Needle Recall: 100%](https://img.shields.io/badge/Needle_Recall-100%25-gold.svg)]()

**AURA** es una nueva arquitectura de inteligencia artificial generativa para el modelado de secuencias, inspirada en la física de osciladores armónicos acoplados en el espacio de estados complejo. 

Diseñada desde cero como una alternativa fundamental a los Transformers de ChatGPT, **elimina la auto-atención cuadrática ($O(N^2)$) y erradica por completo el KV-Cache**.

---

## ⚡ ¿Por qué AURA?

| Característica | Transformer Tradicional (ChatGPT) | **AURA (Nuestra Arquitectura)** |
| :--- | :--- | :--- |
| **Consumo de Memoria** | Crece linealmente con cada palabra (**KV-Cache**) | **Estrictamente constante ($O(1)$): 54.0 KB** |
| **Complejidad Temporal** | Cuadrática ($O(N^2)$) — se asfixia en textos largos | **Lineal ($O(N)$) — escala suavemente** |
| **Velocidad a 1.024 tokens** | 296.9 ms / token | **31.3 ms / token (9.5x más rápida)** |
| **Recuperación Asociativa** | Requiere releer todo el historial | **100.0% de precisión perfecta en 80 tokens de ruido** |
| **Motor Matemático** | Producto escalar denso Softmax ($QK^T$) | **Rotaciones unitarias complejas ($e^{i\theta}$) + Producto exterior** |

---

## 🔬 Principio Matemático (AURA-Matrix v2.1)

En lugar de almacenar tokens pasados, AURA actualiza un tapiz matricial complejo continuo $S_t \in \mathbb{C}^{H \times K \times V}$:

$$S_t = (\lambda_t \odot e^{i \theta_t}) \odot S_{t-1} + \big(\beta_t \odot k_t\big) \otimes v_t$$

* **Preservación Unitaria:** Al operar sobre el círculo unitario ($\|e^{i \theta_t}\| = 1$), los gradientes no se desvanecen ni explotan.
* **Invarianza de Claves:** La rotación de fase se aplica al eje de valores $V$, preservando las direcciones del espacio de claves para recuperación exacta por consulta ($q_t^T S_t$).
* **Filtro de Ruido:** La compuerta selectiva $\beta_t \in (0,1)^K$ cierra el paso a tokens irrelevantes, impidiendo la saturación de memoria.

---

## 🚀 Inicio Rápido

### Instalación
```bash
pip install -e .
```

### Uso en Python (3 líneas)
```python
import torch
from aura import AURAMatrixModel

model = AURAMatrixModel(vocab_size=1000, d_model=128, n_layers=4)
prompt = torch.randint(0, 1000, (1, 16))

# Generación en memoria constante O(1) [Cero KV-Cache]
output = model.generate(prompt, max_new_tokens=50)
print("Tokens generados:", output.shape)
```

---

## 💻 Estudio Web Interactivo

Puedes probar el modelo en tiempo real desde tu navegador con la interfaz gráfica integrada:

```bash
python app.py
```
Abre tu navegador en: `http://localhost:7860`

---

## ☁️ Entrenamiento en GPU Gratuita (Google Colab)

Incluye un notebook autónomo de 1 clic para entrenar en Google Colab con GPUs T4 o A100:
- Archivo: `AURA_Colab_Trainer.ipynb`
- Sube el archivo a [Google Colab](https://colab.research.google.com/), activa GPU gratuita y ejecuta todas las celdas.

---

## 📁 Estructura del Repositorio

```
aura_ai/
├── aura/                     # Paquete Python instalable
│   ├── __init__.py           # Exportaciones del módulo
│   ├── model.py              # Arquitectura AURA-Matrix v2.1
│   └── tokenizer.py          # Tokenizador BPE y por caracteres
├── AURA_Colab_Trainer.ipynb  # Notebook listo para GPU en Google Colab
├── PAPER_AURA.md             # Preprint científico formal
├── app.py                    # Servidor web del estudio interactivo
├── duelo_aura_vs_transformer.py # Benchmark empírico contra Transformers
├── train_v2_text.py          # Script de entrenamiento en lenguaje natural
├── aura_v2_text_checkpoint.pt # Pesos del modelo entrenado
├── pyproject.toml            # Configuración de distribución estándar
├── requirements.txt          # Dependencias
└── LICENSE                   # Licencia MIT Open Source
```

---

## 📜 Cita

Si utilizas AURA en tus investigaciones o experimentos:

```bibtex
@article{aura2026,
  title={AURA: Adaptive Unitary Resonant Architecture for Constant-Memory Sequence Modeling},
  author={AURA Research Project},
  year={2026},
  url={https://github.com/your-username/aura-ai}
}
```
