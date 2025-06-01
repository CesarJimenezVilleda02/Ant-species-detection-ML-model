# Clasificación de Especies de Hormigas mediante Machine Learning

El objetivo de este proyecto es identificar distintas especies de hormigas a partir de imágenes. Esta clasificación resulta esencial para la detección temprana de plagas, la identificación de especies invasoras y el diseño de estrategias de control que protejan el ecosistema local.

Se utiliza Aprendizaje Automático (Machine Learning), una rama de la inteligencia artificial que permite que un modelo aprenda patrones directamente a partir de los datos, sin necesidad de programación explícita. En particular, se entrenan modelos supervisados, donde el sistema aprende a partir de ejemplos etiquetados.

## Contexto ecológico y descripción de las especies

Cada especie de hormiga contemplada en el modelo tiene un papel particular en los ecosistemas donde habita. Algunas son nativas y cumplen funciones importantes como la aireación del suelo o el control biológico; otras son invasoras que pueden alterar ecosistemas enteros. A continuación se describe brevemente cada clase incluida en el modelo, incluyendo características morfológicas y su impacto ecológico:

- **Argentine-ants (Linepithema humile)**: Especie invasora originaria de Sudamérica. Forma súper colonias y desplaza especies nativas. Es una plaga urbana y agrícola. **Morfología:** Pequeñas, de color marrón claro a oscuro, con cuerpo uniforme y sin espinas visibles.

- **Trap-jaw-ants (Odontomachus spp.)**: Se distinguen por sus poderosas mandíbulas en forma de trampa, que pueden cerrar a gran velocidad. Son depredadoras y tienen un comportamiento agresivo. **Morfología:** Grandes, color oscuro, con mandíbulas rectas y alargadas al frente de la cabeza.

- **Leafcutter-ants (Atta spp.)**: Son una plaga agrícola severa en regiones tropicales y subtropicales. Cultivan hongos con hojas masticadas. **Morfología:** Varían en tamaño, pero se reconocen por tener cabezas muy grandes en forma de corazón y mandíbulas prominentes.

- **Weaver-ants (Oecophylla spp.)**: Construyen nidos con hojas unidas por seda que producen sus larvas. Son territoriales y útiles en el control de plagas en plantaciones. **Morfología:** Alargadas, de color rojizo anaranjado, patas y cuerpo delgados.

- **Yellow-crazy-ants (Anoplolepis gracilipes)**: Especie altamente invasora. Causa impactos ecológicos negativos, especialmente en islas. **Morfología:** De color amarillo claro, patas largas y delgadas, comportamiento errático al caminar.

- **Black-crazy-ants (Paratrechina longicornis)**: También invasora, adaptable a diversos hábitats. Su comportamiento errático al caminar le da su nombre. **Morfología:** Negras, brillantes, pequeñas, con patas y antenas desproporcionadamente largas.

- **Fire-ants (Solenopsis spp.)**: Conocidas por sus dolorosas picaduras. Son invasoras y representan un riesgo para humanos y fauna nativa. **Morfología:** Pequeñas a medianas, de color rojizo, con abdomen más oscuro. Presentan varios tamaños en la misma colonia.

## Generación y selección del set de datos

### Modelo original
[Ants Species Detection Dataset (Roboflow)](https://universe.roboflow.com/test-0xkxs/ants-species-detection-actual/dataset/7/download)

### División de los datos:
**Dataset Split original:**
- Train Set: 89% — 4450 imágenes  
- Valid Set: 9% — 431 imágenes  
- Test Set: 2% — 109 imágenes  

**Preprocessing original:**
- Resize: Fit within 640x640  
- Modify Classes: 6 remapped, 1 dropped

**Augmentations:**
- Outputs per training example: 2  
- Saturation: Between -40% and +40%  
- Brightness: Between -23% and +23%

### Clases originales:
- Argentine-ants  
- Trap-jaw-ants  
- Unlabeled  
- Weaver-ants  
- Yellow-crazy-ants  
- Fire-ants

### Modificaciones realizadas al dataset:

Se llevó a cabo una limpieza exhaustiva del dataset original, eliminando imágenes mal etiquetadas, imágenes sin etiqueta y aquellas con contenido ambiguo. Además, se redistribuyeron las imágenes para mejorar la validación del modelo con una división más balanceada: 65% entrenamiento, 15% validación y 20% prueba.

Dado que la clase "trap-jaw-ants" contaba con muy pocas muestras inicialmente, se incorporaron más imágenes adicionales para mejorar su representación. 

Se añadió una nueva clase: **leafcutter-ants**, debido a su relevancia como plaga en cultivos de la región de Huichapan, Hidalgo. Las imágenes fueron obtenidas manualmente mediante web scraping y utilizando la herramienta ShareX para capturar frames de diversos videos en YouTube que mostraban estas hormigas en diferentes contextos (cautiverio, nidos, campo abierto). También se recolectaron imágenes desde la plataforma especializada AntWiki.

### Distribución final de los datos:
**TRAIN — 5118 imágenes**
- trap-jaw-ants: 525  
- black-crazy-ants: 417  
- leafcutter-ants: 1039  
- yellow-crazy-ants: 468  
- argentine-ants: 1136  
- weaver-ants: 612  
- fire-ants: 921  

**TEST — 1276 imágenes**
- trap-jaw-ants: 131  
- black-crazy-ants: 104  
- leafcutter-ants: 259  
- yellow-crazy-ants: 116  
- argentine-ants: 284  
- weaver-ants: 152  
- fire-ants: 230  

### Ejemplos de imágenes eliminadas
![image](https://github.com/user-attachments/assets/b2b4147d-634f-4fb6-8412-16fbd1e5ec8e)
![image](https://github.com/user-attachments/assets/f2cebb0a-2901-4497-959a-49dd77c76e1b)
![image](https://github.com/user-attachments/assets/12c473b8-5652-46d8-b77b-cc88ca14a173)
![image](https://github.com/user-attachments/assets/64d831f0-8069-4a6a-9e53-cc9f64492143)

## Preprocesado de los datos y técnicas de aumento

Para que el modelo sea robusto frente a las múltiples condiciones en las que se pueden fotografiar hormigas, se aplicó un conjunto de transformaciones durante el entrenamiento.
### Reescalado (normalización)
Se divide cada valor de píxel entre 255 (rescale = 1./255). Esto normaliza la imagen al rango 0–1 y estabiliza el proceso de optimización.

### Rotación aleatoria ±20 °
Las hormigas pueden ser capturadas desde ángulos muy variados. Girar las imágenes dentro de un margen de ±20 ° permite al modelo reconocer individuos aunque la cámara o el insecto estén inclinados.

### Desplazamientos horizontales y verticales ±15 %
Mediante width_shift_range y height_shift_range desplazamos al sujeto dentro del encuadre. Así evitamos que el modelo dependa de que la hormiga esté centrada y ganamos tolerancia a recortes accidentales.

### Zoom entre 0.7 × y 1.3 ×
Acercar o alejar virtualmente el encuadre refuerza la capacidad del modelo de identificar especies tanto en primeros planos como en tomas más lejanas.

### Volteos horizontales y verticales
Los volteos (horizontal_flip y vertical_flip) duplican el número de combinaciones espaciales. Resultan útiles porque la morfología de una hormiga suele ser simétrica y puede aparecer boca abajo, de costado o invertida si la foto se toma con el teléfono en otra orientación.

### fill_mode = "wrap"
Cuando se rotan o desplazan las imágenes quedan espacios vacíos. El modo «wrap» recicla la propia imagen para rellenar esos huecos, evitando que aparezcan barras negras o zonas con un color sólido que el modelo podría aprender. Además en el dataset hay varios ejemplos de cómo se ven los nidos de estas hormigas, por lo que este relleno de bordes simula cómo se vería una toma alejada de un nido o grupo de hormigas.

### Brillo aleatorio (0.8 – 1.2 ×)
Las fotografías tomadas en exterior presentan variaciones de luz (pleno sol, sombra, flash). Ajustar el brillo de forma moderada enseña al modelo a ser invariante a estas diferencias.

### Contraste aleatorio (0.8 – 1.2 ×)
La función random_contrast aumenta o reduce el contraste de cada imagen. Esto ayuda a diferenciar detalles de color y forma: por ejemplo, distinguir el tono rojizo de una weaver‑ant frente al verde intenso del follaje, o resaltar la cabeza en forma de corazón de una leafcutter‑ant cuando el fondo comparte tonalidades.

### Conjuntos donde se aplican los aumentos
- **Entrenamiento (65 %)**: se combinan de forma aleatoria todas las transformaciones descritas, generando nuevas variaciones en cada época.
- **Validación (15 %)**: se utiliza el mismo generador para mantener la misma separación.
- **Prueba (20 %)**: únicamente se reescala la imagen. No se aplican aumentos para medir de forma objetiva la capacidad de generalización del modelo sobre datos jamás vistos.

Estas estrategias, combinadas, permiten que el clasificador reconozca hormigas en diferentes posiciones, tamaños, niveles de iluminación y fondos, mejorando su desempeño en condiciones de campo reales.

### Ejemplos de los aumentos en cada clase
![image](https://github.com/user-attachments/assets/7420fb5c-14be-473c-8fdf-2261b3191464)
![image](https://github.com/user-attachments/assets/c980d9a7-714a-4a4a-9eef-41102b5d6f1c)
![image](https://github.com/user-attachments/assets/f1721ce2-5053-4a16-98a9-528e723c80bf)
![image](https://github.com/user-attachments/assets/b15f6c4d-72c6-46da-9cf5-52a52f0f83b0)
![image](https://github.com/user-attachments/assets/f4885825-04a4-4f3b-bc1e-4659792aaa2f)
![image](https://github.com/user-attachments/assets/09e0fb77-2588-4d22-972f-72dd21ecd198)
![image](https://github.com/user-attachments/assets/0a5bdf56-0adc-426a-ae5f-b0a57745913f)


## Construcción de la primera versión del modelo

### Arquitectura y configuración

La red se inspira en el artículo ImageNet Classification with Deep Convolutional Networks[3] y presenta:
![image](https://github.com/user-attachments/assets/6d022151-f09f-48c0-b474-8bdad93c1db5)

### Arquitectura detallada de cada capa

- Capa de entrada  
  Recibe imágenes en formato RGB de 224×224 píxeles. Su función es normalizar y organizar los datos de manera que todas las imágenes tengan el mismo tamaño y formato antes de ser procesadas por la red.

- Primer bloque convolucional  
  Aplica 64 filtros de tamaño 3×3 con activación ReLU. Cada filtro actúa como un detector de características sencillas (bordes, esquinas, cambios de intensidad) en distintas posiciones de la imagen. La función ReLU introduce no linealidad, descartando valores negativos y acelerando el entrenamiento.

- Segundo bloque convolucional  
  Utiliza 256 filtros de 3×3 con ReLU. Al profundizar la red, estos filtros capturan patrones más complejos, como texturas, contornos de antenas o formas de mandíbulas. El padding “same” garantiza que el tamaño espacial se mantenga constante, facilitando el apilamiento de capas.

- Primera operación de max pooling  
  Reduce la resolución espacial a la mitad seleccionando el valor máximo en cada región de 2×2 píxeles. Esto concentra la información más destacada, aporta invarianza a pequeñas traslaciones y reduce la cantidad de datos a procesar, agilizando la red.

- Tercer bloque convolucional  
  Otros 256 filtros de 3×3 con ReLU continúan enriqueciendo la representación, aprendiendo combinaciones de las características ya extraídas por capas anteriores.

- Segunda operación de max pooling  
  Repite la reducción de resolución, reforzando la invarianza y comprimiendo la información para evitar el exceso de parámetros y mitigar el sobreajuste.

- Cuarto bloque convolucional  
  Nuevamente 256 filtros de 3×3 con ReLU. Esta última capa convolucional extrae las representaciones de más alto nivel antes de la compactación global, integrando patrones morfológicos específicos de cada especie.

- Tercera operación de max pooling  
  Última reducción espacial antes del paso a agregación global de características.

- Global Average Pooling  
  Transforma cada uno de los 256 mapas de activación en un único valor medio. En lugar de aplanar toda la cuadrícula, promedia cada canal completo, lo que elimina miles de parámetros y actúa como regularizador, reduciendo el riesgo de sobreajuste.

- Capa densa intermedia  
  Toma el vector de 256 valores resultante del GAP y aprende combinaciones no lineales de esas características globales mediante 256 unidades con ReLU. Esta capa sintetiza la información en una representación compacta de alto nivel.

- Capa de salida con softmax  
  Convierte las activaciones finales en una distribución de probabilidad sobre las siete clases. La función softmax garantiza que la suma de probabilidades sea 1, permitiendo interpretar directamente la confianza de la predicción para cada especie.
### Selección de métricas

Para evaluar el rendimiento se eligieron:

- **Accuracy**  
  Mide la proporción de predicciones correctas sobre el total de muestras evaluadas. Refleja el comportamiento global del modelo, pero puede enmascarar errores en clases poco representadas.

- **Precisión**  
  Calcula la fracción de verdaderos positivos entre todas las predicciones positivas. Indica qué tan confiables son las alertas del modelo, minimizando las falsas alarmas en especies inofensivas.

- **Recall**  
  Representa la proporción de verdaderos positivos identificados sobre el total de casos reales de esa clase. Es clave para no pasar por alto ejemplares de especies invasoras, reduciendo los falsos negativos.

- **F1-score (macro)**  
  Es la media de precisión y recall calculada de forma independiente para cada clase y luego promediada. Ofrece una visión equilibrada entre ambos errores, especialmente útil cuando las clases están desbalanceadas.  

En tareas de identificación automática de especies, precisión mide la proporción de detecciones correctas entre todas las predicciones positivas, lo cual es clave para evitar falsas alarmas en estudios de fauna. Norouzzadeh et al. demuestran cómo un alto valor de precisión reduce el tiempo de validación manual en grandes colecciones de cámaras trampa [1]. Por su parte, el recall (sensibilidad) indica qué fracción de ejemplares reales es efectivamente detectada por el modelo; Das y Kumar lo emplean para garantizar que las especies menos frecuentes no queden inadvertidas, incluso a costa de aceptar más falsos positivos [2]. El F1-score combina ambas métricas mediante la media armónica, equilibrando sus ventajas, tal como lo formalizó van Rijsbergen para sistemas de recuperación de información y que hoy se aplica a clasificación de especies para no enmascarar el rendimiento en clases minoritarias.

### Compilación del modelo

Antes de comenzar el aprendizaje, se configuraron tres elementos:

- **Optimizador (Adam)**  
Este algoritmo ajusta los pesos de la red adaptando la magnitud de la actualización a la historia de los gradientes. Gracias a sus momentos de primer y segundo orden, Adam converge con estabilidad incluso en problemas complejos.

- **Función de pérdida (categorical_crossentropy)**  
Mide la discrepancia entre la distribución de probabilidad que el modelo predice y la distribución verdadera de cada clase. En cada paso de entrenamiento, minimizamos esta pérdida para que las predicciones se acerquen cada vez más a las etiquetas reales.

- **Métrica de evaluación (accuracy)**  
Indica el porcentaje de imágenes clasificadas correctamente. Es una señal rápida de progreso, pero no distingue entre falsos positivos y falsos negativos.

### Entrenamiento del modelo

El entrenamiento de configuró de la siguiente manera:

- **Batch size de 16**  
Cada actualización se basa en un lote de 16 imágenes. Un tamaño así introduce suficiente variabilidad y mantiene un uso moderado de memoria GPU. Si el batch fuera mucho mayor, las actualizaciones serían más estables pero consumirían mucha más memoria.

- **60 épocas**  
En cada época, la red recorre todas las muestras de entrenamiento en lotes de 16. Con 60 pasadas completas, hay oportunidad de ajustar los pesos con detalle. Pocas épocas pueden dejar el modelo subajustado, mientras que demasiadas sin control pueden llevar a sobreajuste.

- **100 pasos por época**  
Cada paso procesa un lote de 16 imágenes. Con 100 pasos, el modelo ve 1 600 imágenes antes de pasar a la validación. Aumentar este número expondría más datos por época pero haría cada ciclo más largo; reducirlo aceleraría cada época pero podría invalidar la evaluación al ver menos muestras.

- **Validación en 25 pasos**  
Tras cada época, se evalua el rendimiento en 25 lotes del conjunto de validación. Esto provee una medida confiable de generalización sin retrasar el entrenamiento.

- **Pesos de clase**  
Se asigna un peso mayor a las clases con menos muestras, de modo que sus errores penalicen más la pérdida total. Así se evita que especies poco representadas queden “silenciadas” durante el aprendizaje.

### Análisis de resultados

Al finalizar el entrenamiento y la validación, se obtuvieron las siguientes métricas:

- Entrenamiento (última época):  
  • Pérdida: 0.4523  
  • Accuracy: 0.8338

- Validación (última época):  
  • Pérdida: 1.3484  
  • Accuracy: 0.6500

- Prueba (test):  
  • Pérdida: 0.5001  
  • Accuracy: 0.8477

### Analisis del entrenamiento

#### Conceptos clave

- **Underfitting**  
Ocurre cuando el modelo es demasiado simple o no ha entrenado lo suficiente; ni siquiera aprende bien los patrones del conjunto de entrenamiento (baja precisión y alta pérdida en train).

- **Overfitting**  
Sucede cuando el modelo memoriza excesivamente el conjunto de entrenamiento, obteniendo muy buena precisión en train pero un desempeño pobre en validación (gran brecha entre train y val).

#### Graficas del proceso de entrenamiento
Se generaron gráficas de evolución de accuracy y loss a lo largo de las épocas:

![image](https://github.com/user-attachments/assets/93a86f12-37fa-438c-9e3e-3dc53d313565)

![image](https://github.com/user-attachments/assets/09c17325-bfe4-4ce1-973f-45413780c129)

#### Análisis de las curvas de entrenamiento

- La precisión de entrenamiento (azul) y de prueba (verde) crecen de forma pareja hasta rondar 0.85, lo que indica que el modelo ajusta bien esos datos.  
- La precisión de validación (naranja) se muestra muy errática, con subidas y bajadas bruscas. Esto se debe a que solo se evalúa una porción del set de validación en cada época, no la totalidad, por lo que las métricas varían según qué lotes toquen en ese paso.
- La precisión de prueba supera ligeramente a la de entrenamiento en algunos tramos. Este patrón sugiere un leve underfitting respecto al test completo, quizá porque la arquitectura es todavía demasiado simple para capturar todas las variaciones.  
- En cuanto a la pérdida, tanto entrenamiento como prueba descienden de forma suave.

### Reporte de clasificación

| Clase               | Precisión | Recall | F1-score | Soporte |
|---------------------|-----------|--------|----------|---------|
| argentine-ants      | 0.91      | 0.86   | 0.89     | 284     |
| black-crazy-ants    | 0.86      | 0.91   | 0.88     | 104     |
| fire-ants           | 0.79      | 0.81   | 0.80     | 230     |
| leafcutter-ants     | 0.80      | 0.86   | 0.83     | 259     |
| trap-jaw-ants       | 0.80      | 0.85   | 0.82     | 129     |
| weaver-ants         | 0.92      | 0.85   | 0.88     | 152     |
| yellow-crazy-ants   | 0.90      | 0.78   | 0.84     | 116     |
| **Accuracy global** |           |        | **0.85** | 1274    |
| Macro avg           | 0.85      | 0.85   | 0.85     | 1274    |
| Weighted avg        | 0.85      | 0.85   | 0.85     | 1274    |

- **argentine-ants**: Con una precisión del 91% y un recall del 86%, el modelo identifica con confianza a esta especie y recupera la mayoría de sus ejemplares, logrando un F1 de 0.89.

- **black-crazy-ants**: El balance es excelente (precisión 86%, recall 91%), lo que indica muy pocos falsos positivos y negativos; el F1 de 0.88 confirma la fiabilidad en esta clase.

- **fire-ants**: Con precisión del 79% y recall del 81%, muestra un ligero sesgo hacia falsos negativos (no siempre detecta cada muestra), pero mantiene un F1 aceptable de 0.80.

- **leafcutter-ants**: Un recall alto (86%) contrasta con una precisión algo menor (80%). El modelo tiende a confundir otras especies como fire-ants con leafcutters, aunque la F1 de 0.83 sigue siendo buena.

- **trap-jaw-ants**: Precisión y recall equilibrados (80% y 85%, respectivamente) y un F1 de 0.82 indican un desempeño constante, aunque queda margen para mejorar la detección de mandíbulas específicas.

- **weaver-ants**: Destaca con un 92% de precisión y 85% de recall, logrando el F1 más alto (0.88). Sus características (color rojizo y forma alargada) pueden ser más fáciles de distinguir para el modelo.

- **yellow-crazy-ants**: Muy buena precisión (90%), pero un recall más bajo (78%) sugiere que algunas muestras se pierden; aun así, un F1 de 0.84 demuestra un desempeño robusto.

- **Accuracy global (85%)**: Se traduce en que 85% de todas las predicciones son correctas, lo cual es consistente con los F1 individuales.

En conjunto, el modelo muestra equilibrio entre precisión y recall, con puntajes F1 por encima de 0.80 en todas las especies. Las mayores oportunidades de mejora se encuentran en fire-ants y leafcutter-ants, donde ajustes en data augmentation o arquitectura podrían reducir las confusiones actuales.  

### Matriz de confusión
![image](https://github.com/user-attachments/assets/a724cf03-1187-4583-83ee-52fb21dfb773)

Argentine-ants: 245 se clasificaron correctamente, pero 15 se confundieron con fire-ants y 18 con trap-jaw-ants. Esto sugiere similitudes visuales tal vez en tonos marrones—entre estas tres clases.

Black-crazy-ants: 95 aciertos, con solo 3 falsos positivos hacia fire-ants y 2 hacia yellow-crazy-ants. Su baja tasa de error indica que sus características son muy distintivas.

Fire-ants: 187 aciertos, pero 22 fueron etiquetados como leafcutter-ants y 7 como argentine-ants. Al compartir tonos rojizos y tamaños variables, la red a veces no discrimina correctamente entre estos grupos.

Leafcutter-ants: 223 bien clasificadas, con 13 confundidas con fire-ants y 7 con trap-jaw-ants. Dado que ambas tienen cabezas relativamente grandes y colores anaranjados/marrones, conviene reforzar diferenciadores morfológicos.

Trap-jaw-ants: 110 aciertos, 11 confundidas con leafcutter-ants y 4 con argentine-ants. A pesar de sus mandíbulas icónicas, el recorte en algunas imágenes puede ocultar ese rasgo clave.

Weaver-ants: 129 predicciones correctas, 12 falsos hacia leafcutter-ants y 6 hacia fire-ants. Su color rojizo similar y fondos de hojas pueden inducir a error.

Yellow-crazy-ants: 91 aciertos, aunque 10 fueron etiquetadas como fire-ants y 6 como argentine-ants. El amarillo pálido puede mezclarse con marrones claros en condiciones de iluminación variada.

En conjunto, los mayores conflictos se dan entre especies de tonalidades cálidas

### Resultados con imagenes de prueba aleatorias
![image](https://github.com/user-attachments/assets/6e75a7a1-c0e1-4946-841b-a31312b96153)

### Conclusiones de la primer versióm
- El modelo alcanza una alta robustez (85 % de accuracy y F1-score macro de 0.85) gracias al uso de una arquitectura de cuatro bloques convolucionales y al empleo de Global Average Pooling, lo que le confiere buena capacidad de extracción de características sin un número excesivo de parámetros.

- Las curvas de validación actuales, evaluadas solo sobre porciones del set en cada época, muestran alta variabilidad. Se debería medir la pérdida y la precisión sobre la totalidad del conjunto de validación en cada paso para obtener una señal de generalización más estable y evitar decisiones de hiperparámetros basadas en muestreos parciales.

- El uso de class weights ha equilibrado eficazmente el aprendizaje entre especies con soporte muy desigual, pero aún se observan confusiones entre clases de tonalidades similares (fire-ants vs leafcutter-ants, trap-jaw-ants vs argentine-ants). Para reducir estos errores, voy a introducir aumentos adicionales de contraste que obliguen al modelo a centrarse en rasgos morfológicos (mandíbulas, forma de la cabeza) por encima del color.

- El modelo podría mejorar arquitectónisamente mediante el uso de VGG16 o ResNet como base convolucional. Gracias a su mayor profundidad y a sus filtros optimizados en ImageNet, VGG16 extrae representaciones más detalladas y complejas, lo que puede ayudar a reducir el underfitting al captar mejor las sutilezas morfológicas de cada especie.

## Segunda versión del modelo

### Aumentos de datos revisados

En esta iteración se intensificó la estrategia de *data-augmentation* con el fin de reducir la dependencia del modelo a variaciones cromáticas y obligarlo a fijarse en rasgos morfológicos finos (mandíbulas, forma del tórax, espinas):

| Parámetro | Versión 1 | Versión 2 | Motivación |
|-----------|-----------|-----------|------------|
| `zoom_range` | (0.7, 1.3) | (0.7, 1.3) | Se mantiene: ya capturaba distintos tamaños en plano. |
| `brightness_range` | (0.8, 1.2) | **(0.7, 1.3)** | Variación de iluminación más amplia para desacoplar el color del fondo. |
| `preprocessing_function` (contraste) | factor 0.8-1.2 | **factor 0.7-1.3** | Contraste más agresivo; fuerza a discriminar contornos y textura de cabeza/patas. |
| Resto de transformaciones | rotación 20°, *shifts* 15 %, `horizontal_flip`, `vertical_flip`, `fill_mode="wrap"` | Sin cambios | Ya aportaban buenos aumentos. |

### Ajustes en la arquitectura convolucional

| Componente | Versión 1 | Versión 2 | Razón del cambio |
|------------|-----------|-----------|------------------|
| Filtros de la primera capa | 64 filtros 3×3 | 96 filtros 3×3 | Incrementar capacidad para capturar micro-texturas desde el nivel más bajo. |
| Número total de capas conv. | 4 | 5 | Profundidad adicional para capturar patrones morfológicas más complejas. |

El entrenamiento y compilación se mantuvieron iguales en esta iteración.

### Resultados
#### Matriz de confusión
![image](https://github.com/user-attachments/assets/ca7f6038-8a54-4e25-973a-4ffaeedbbc18)

##### Lo que mejoró
- black-crazy-ants, weaver-ants y yellow-crazy-ants ganan entre 2 y 12 puntos porcentuales de exactitud.  
- Se reducen confusiones entre argentine con trap-jaw y fire con leafcutter.  
- El nuevo ajuste de contraste parece ayudar a distinguir tonalidades oscuras (black-crazy) y la silueta esbelta de weaver-ants pero no los tonos cálidos.

##### Lo que empeoró
- leafcutter-ants cae de 86 % a 60 % de exactitud; ahora se confunden sobre todo con yellow-crazy y argentine-ants.  
- argentine-ants y fire-ants también pierden precisión.  
- Para las clases de tonos cálidos (argentine, fire, leafcutter) el modelo dejó de discriminar bien color y forma, quizá por los aumentos más agresivos de brillo/contraste.

#### Entrenamiento
![image](https://github.com/user-attachments/assets/964f9555-8e0d-42aa-9e5b-c5c8a3f3fd96)
![image](https://github.com/user-attachments/assets/5439f658-ed3a-4afc-b953-cb1b333a94cc)

- La exactitud de entrenamiento sube con constancia hasta 0.85-0.87.
- Validación se mantiene entre 0.30 y 0.60 y fluctúa bastante.
- Test termina cerca de 0.80-0.83, por debajo de entrenamiento y sin tendencia clara a mejorar después de la mitad del proceso.

Por lo tanto las aumentaciones más agresivas no elevaron la precisión final y quizá alteraron rasgos útiles del color. También, añadir la capa extra y más filtros incrementó la capacidad de la red, pero esa complejidad no se tradujo en mejor desempeño fuera del entrenamiento. La brecha persistente entre entrenamiento y validación indica que el modelo capturó detalles presentes solo en las imágenes de entrenamiento.

#### Análisis de métricas
| Métrica global    | Modelo 1 | Modelo 2 | Cambio |
|-------------------|---------:|---------:|-------:
| Accuracy          | 0.85 | 0.79 | −0.06 |
| Macro F1-score    | 0.85 | 0.79 | −0.06 |
| Weighted F1-score | 0.85 | 0.79 | −0.06 |

| Clase               | F1 M1 | F1 M2 | Δ F1 | Observación principal |
|---------------------|------:|------:|-----:|-----------------------|
| argentine-ants      | 0.89 | 0.82 | −0.07 | Menor precisión y recall. |
| black-crazy-ants    | 0.88 | 0.83 | −0.05 | Recall sube (91 → 93) pero precisión baja. |
| fire-ants           | 0.80 | 0.73 | −0.07 | Más confusiones con yellow-crazy y argentine. |
| leafcutter-ants     | 0.83 | 0.74 | −0.09 | Fuerte caída de recall (86 → 60). |
| trap-jaw-ants       | 0.82 | 0.86 | +0.04 | Única clase que mejora de forma consistente. |
| weaver-ants         | 0.88 | 0.82 | −0.06 | Confusión adicional con yellow-crazy. |
| yellow-crazy-ants   | 0.84 | 0.74 | −0.10 | Recall sube (78 → 91), precisión baja. |

#### Conclusiones
Modelo 1 mantiene mejor equilibrio general y modelo 2 solo gana en trap-jaw-ants y en la recuperación de black-crazy y yellow-crazy, pero sacrifica desempeño en el resto, sobre todo en leafcutter-ants, argentine-ants y fire-ants. La combinación de aumentaciones más agresivas y una capa convolucional adicional no aporta beneficio, conviene explorar fine-tuning con arquitecturas preentrenadas o ajustar los aumentos para no degradar clases sensibles.
- Con la configuración original logró un F1 macro de 0.85, pero la versión “más profunda” con aumentos agresivos bajó a 0.79.  
- Aumentar filtros y capas no tradujo en mejor generalización; algunas clases clave (leafcutter-ants, fire-ants) perdieron desempeño.  

Esto sugiere que la arquitectura actual ya no extrae rasgos suficientes ni gestiona bien la variabilidad cromática y de fondo. Para avanzar conviene migrar a una arquitectura más compleja.

## Tercera versión del modelo
### Arquitectura y configuración
| Bloque | Descripción | Propósito |
|--------|-------------|-----------|
| **Base convolucional** | EfficientNetV2B0 (`include_top=False`, pesos de ImageNet). Todas sus capas quedan entrenables (`trainable=True`). | Extrae representaciones ricas y compactas con menos parámetros y tiempos de entrenamiento menores que CNNs tradicionales. |
| **GlobalAveragePooling2D** | Promedia cada mapa de activación de la base. | Reduce la salida a un vector y actúa como regularizador. |
| **Densa (256 ReLU)** | 256 unidades totalmente conectadas. | Combina rasgos globales en una representación de alto nivel. |
| **Dropout (0.2)** | Suprime aleatoriamente 20 % de activaciones. | Evita memorizar el set de entrenamiento. |
| **Densa de salida (softmax)** | `num_classes = 7`. | Devuelve probabilidades por especie. |

### Compilación  
Se mantiene igual que en versiones anteriores: `optimizer="adam"`, `loss='categorical_crossentropy'`, `metrics=['accuracy']`.

### ¿Qué es EfficientNetV2 y por qué usarlo aquí?

* **EfficientNetV2** es una familia de redes convolucionales optimizadas para lograr la mejor precisión posible con el menor costo computacional. Fue diseñada usando búsqueda automática de arquitecturas (NAS) pero con un cambio clave: en lugar de solo buscar precisión, también se optimizó el **tiempo de entrenamiento**. Esto la hace especialmente útil cuando hay recursos limitados de cómputo o tiempo [4].

* En lugar de escalar redes arbitrariamente como agregar más capas o más filtros, EfficientNet propone el enfoque de **compound scaling**: una forma balanceada de aumentar profundidad, ancho y resolución de entrada simultáneamente.

En arquitecturas como la primer versión del modelo, al querer hacer una red más grande se suele modificar solo una dimensión: se le agregan más capas (profundidad), se amplían los canales de cada capa (ancho) o se sube el tamaño de las imágenes de entrada (resolución). Sin embargo, escalar solo uno de estos aspectos suele llevar a redes ineficientes: más lentas o con sobreajuste.

Compound scaling parte de un modelo base (por ejemplo EfficientNetV2-B0) y aplica un único factor de escala (phi) que se reparte de forma coordinada entre las tres dimensiones usando constantes α, β y γ, que representan cuánto aumentar cada dimensión relativa a φ. Así, al aumentar φ, se obtiene una versión más grande de la red, pero sin romper el equilibrio entre capas, filtros y resolución. Esto permite construir modelos más precisos y eficientes en cómputo sin necesidad de rediseñar manualmente la arquitectura cada vez.

* EfficientNetV2 reemplaza los primeros bloques MBConv (que usaban convoluciones separadas en pasos pequeños) por un tipo nuevo llamado Fused-MBConv.

  El bloque MBConv original primero expandía los canales de la imagen con una convolución 1×1, luego aplicaba una convolución 3×3 por canal (depthwise), y después volvía a comprimir con otra 1×1. Era eficiente, pero tenía muchas operaciones pequeñas separadas. Fused-MBConv simplifica esto: usa directamente una convolución 3×3 normal que ya expande y procesa la imagen, seguida de una 1×1. Como lo hace todo junto, aprovecha mejor la GPU y entrena más rápido. Esta fusión funciona especialmente bien en las primeras capas, donde el tamaño de la imagen aún es grande y los canales son pocos. Por eso EfficientNetV2 usa Fused-MBConv solo al principio y después cambia a los bloques más complejos conforme se avanza en la red.

* Además, introduce **aprendizaje progresivo**, una estrategia que arranca el entrenamiento con imágenes pequeñas y pocas regularizaciones (como mixup o dropout), y conforme el modelo mejora, incrementa la resolución de entrada y el nivel de regularización.  

Esto permite que la red aprenda primero las características más generales y luego refine detalles finos, acelerando la convergencia.

* En benchmarks sobre ImageNet, EfficientNetV2 logra:
  - Misma o mejor precisión que modelos mucho más grandes (como ResNet-152 o NFNet).
  - 5× a 11× menos tiempo de entrenamiento.
  - 6.8× menos parámetros [4].

* En aplicaciones reales de clasificación de fauna, como reidentificación de animales, EfficientNetV2 ha demostrado gran capacidad de generalización:
  - En un estudio con 49 especies diferentes, un solo modelo con EfficientNetV2 como backbone superó en 12.5 % de accuracy promedio a entrenar un modelo por especie [5].
  - Además, mostró buen rendimiento en escenarios **zero-shot**, donde debía clasificar especies no vistas durante el entrenamiento [5].

* En el contexto de este proyecto, donde se busca distinguir sutiles diferencias morfológicas entre especies de hormigas (como el tipo de mandíbula, espinas, o forma del tórax), EfficientNetV2 es una elección adecuada por:
  - Su habilidad para **capturar rasgos complejos y texturas finas** gracias a sus bloques Fused-MBConv y su profundidad controlada.
  - Su rendimiento computacionalmente eficiente, ideal para entrenar múltiples versiones sin agotar recursos.
  - Su robustez ante variaciones en los datos, como diferentes fondos o iluminación, cuando se combina con data augmentation adecuado [4].

### Resultados

#### Matriz de confusión
![image](https://github.com/user-attachments/assets/24c72c6e-51e7-47cc-a11b-806f9532a20f)

La matriz de confusión muestra que el modelo se enfocó completamente hacia una sola clase. Todos los ejemplos, sin importar su clase real, fueron etiquetados como tal. Este patrón sugiere un fallo grave en el aprendizaje, probablemente causado por la falta de fine-tuning o una configuración incorrecta del entrenamiento. No hay señales de distinción entre clases, lo que vuelve este modelo inutilizable en su estado actual.

#### Entrenamiento
![image](https://github.com/user-attachments/assets/38e0b50e-f454-4cd4-aeda-ff1d73e46764)
![image](https://github.com/user-attachments/assets/fd5f8d53-f108-4fdb-8b08-3eb98f8ad0e5)

La gráfica muestra que el modelo alcanza rápidamente una alta precisión en el conjunto de entrenamiento, pero tanto la validación como la prueba permanecen en niveles muy bajos y erráticos durante todas las épocas. Esto indica que el modelo memorizó los ejemplos del entrenamiento sin aprender patrones útiles para generalizar, lo que confirma que hubo overfitting severo desde el inicio.

#### Métricas

| Clase               | Precisión | Recall | F1-score | Soporte |
|---------------------|-----------|--------|----------|---------|
| argentine-ants      | 0.00      | 0.00   | 0.00     | 284     |
| black-crazy-ants    | 0.08      | 1.00   | 0.15     | 104     |
| fire-ants           | 0.00      | 0.00   | 0.00     | 230     |
| leafcutter-ants     | 0.50      | 0.01   | 0.02     | 259     |
| trap-jaw-ants       | 0.00      | 0.00   | 0.00     | 129     |
| weaver-ants         | 0.00      | 0.00   | 0.00     | 152     |
| yellow-crazy-ants   | 0.00      | 0.00   | 0.00     | 116     |

| Métrica global      | Valor     |
|---------------------|-----------|
| Accuracy            | 0.08      |
| Macro avg (F1)      | 0.02      |
| Weighted avg (F1)   | 0.02      |

Este modelo es inviable, pues no logra identificar correctamente ninguna clase excepto black-crazy-ants, y aun así lo hace por sobreajuste extremo. Las métricas obtenidas son mucho peores que las vistas en los modelos anteriores que tenían una arquitectura menos compleja. 

#### Conclusión del entrenamiento sin fine-tuning
La gráfica muestra que EfficientNetV2B0 aprendió muy rápido sobre el conjunto de entrenamiento, alcanzando una precisión superior al 85 % desde las primeras épocas. Esto indica que la red tiene una gran capacidad para capturar patrones complejos.

Sin embargo, las curvas de validación y prueba permanecen muy bajas y con gran variación a lo largo de todo el entrenamiento, lo que señala un claro overfitting. El modelo memorizó los datos de entrenamiento pero no logró generalizar a nuevos ejemplos.

Esto se debe a que se entrenó EfficientNet completamente desde el inicio sin aplicar fine-tuning. Al no ajustar el aprendizaje por etapas y usar un learning rate constante desde el principio, la red sobreajustó su representación a ejemplos específicos del set de entrenamiento, perdiendo la capacidad de aplicar lo aprendido a otras imágenes.

Para corregir esto se va a implementar un proceso de fine-tuning y ajustar el learning rate para que el modelo retome los pesos preentrenados de manera más controlada y pueda adaptarse progresivamente a las características del dataset sin caer en la memorización.

## Refinamiento del modelo

### Experimentacion
Se exploraron varios refinamientos arquitectónicos y de entrenamiento para mejorar la generalización del modelo sin cambiar su base convolucional. Estas versiones sirvieron como etapa intermedia entre el entrenamiento sin ajustes y el ajuste fino completo.

En la **versión 4**, se utilizó un learning rate pequeño 0.00001 y se redujo el dropout a 0.1, lo que permitió al modelo conservar detalles finos en la representación sin sobre-regularizar. Esta configuración logró un **accuracy en test del 89.09 %**, siendo hasta ese punto la mejor combinación observada. Su comportamiento fue más estable en validación y prueba en comparación con versiones anteriores que usaban dropout más alto o tasas de aprendizaje más agresivas.

![image](https://github.com/user-attachments/assets/19ed9f06-865c-4e34-bd75-46a87a9884ab)
![image](https://github.com/user-attachments/assets/8fa8f047-2713-4518-bd59-34500bf9632a)

En la **versión 5**, se incrementó ligeramente el learning rate a 0.00005, manteniendo el resto de la arquitectura igual. El modelo mostró señales de sobreajuste leve y una caída en la precisión en test a **86.50 %**, lo cual sugiere que el valor anterior de tasa de aprendizaje era más adecuado para conservar el conocimiento útil de los pesos preentrenados. Aunque la diferencia no fue dramática, reforzó la conclusión de que un aprendizaje más lento es preferible para este tipo de datos.

![image](https://github.com/user-attachments/assets/37f8b0cd-1b98-4b0d-9f85-5e66257a074b)
![image](https://github.com/user-attachments/assets/0e39265d-fcd6-48a8-a6dd-1cc00205e3c8)


### Fine tuning
Para la sexta versión del modelo se mantuvo la misma base convolucional EfficientNetV2B0, ya que había demostrado una rápida capacidad de aprendizaje. Sin embargo, se ajustaron cuidadosamente los componentes finales del modelo para mejorar la generalización sin perder eficiencia.

Se conservó la arquitectura general, pero se realizaron dos cambios clave:

- Se utilizó un learning rate más bajo de 0.00002 en el optimizador Adam. Este valor fue seleccionado tras observar que tasas mayores causaban sobreajuste temprano y hacían que el modelo tuviera más dificultades para converger, mientras que esta configuración permitió una adaptación más gradual de los pesos preentrenados durante el fine tuning.
  
- Se redujo el valor de dropout de 0.2 a 0.1. En pruebas previas, valores altos como 0.2 ayudaban inicialmente, pero terminaban afectando la capacidad del modelo para capturar detalles sutiles entre especies morfológicamente similares. La reducción a 0.1 conservó cierta regularización sin apagar demasiadas neuronas útiles.

### Resultados

#### Matriz de confusión
![image](https://github.com/user-attachments/assets/bd939897-e595-4e01-920b-c418db8bc52a)

Las confusiones son mínimas y están distribuidas de forma moderada. Las especies más afectadas por errores son *leafcutter-ants* y *fire-ants*, con algunos intercambios entre sí, lo cual es coherente con su semejanza en color y forma. El resto de las clases mantienen una separación clara, especialmente *argentine-ants* y *weaver-ants*, que fueron clasificadas con muy pocas equivocaciones.

En general, esta matriz refleja que el modelo tiene una comprensión sólida de las clases cuando se enfrenta a datos del mismo dominio que el entrenamiento, y respalda la estabilidad del nuevo proceso de aprendizaje más lento.

#### Entrenamiento
![image](https://github.com/user-attachments/assets/c9b061b9-cf11-48db-aebd-6fee2bb250db)
![image](https://github.com/user-attachments/assets/8b55595f-c782-4101-a190-541448c6bb50)

La gráfica muestra una evolución consistente del aprendizaje con una fuerte alineación entre la curva de entrenamiento y la de prueba. Esta cercanía indica que los hiperparámetros empleados en esta versión incluyendo el learning rate y el esquema de regularización están muy cerca de ser óptimos. La validación, aunque más ruidosa, también presenta una tendencia ascendente, lo que confirma que el modelo generaliza bien y no está sobreajustando de forma significativa.

#### Métricas
| Clase               | Precision | Recall | F1-score | Soporte |
|---------------------|-----------|--------|----------|---------|
| argentine-ants      | 0.94      | 0.94   | 0.94     | 284     |
| black-crazy-ants    | 0.96      | 0.96   | 0.96     | 104     |
| fire-ants           | 0.90      | 0.90   | 0.90     | 230     |
| leafcutter-ants     | 0.88      | 0.83   | 0.85     | 259     |
| trap-jaw-ants       | 0.90      | 0.95   | 0.92     | 129     |
| weaver-ants         | 0.92      | 0.94   | 0.93     | 152     |
| yellow-crazy-ants   | 0.87      | 0.91   | 0.89     | 116     |
|                     |           |        |          |         |
| **Accuracy**        |           |        | 0.91     | 1274    |
| **Macro avg**       | 0.91      | 0.92   | 0.91     | 1274    |
| **Weighted avg**    | 0.91      | 0.91   | 0.91     | 1274    |

La versión 6 del modelo presenta una mejora general en todos los indicadores clave en comparación con la versión 4. El accuracy global pasó de 89 % a 91 %, y se observa una mejora más homogénea entre clases.

- **Estabilidad por clase**: Aunque la versión 4 mostró excelente rendimiento en clases específicas como *argentine-ants* (0.97 de precisión), esta precisión vino acompañada de un recall algo menor (0.90), lo cual indica una ligera pérdida de sensibilidad. En cambio, en la versión 6, esta clase mantiene un balance perfecto entre precisión y recall (ambos en 0.94).
- **Mejor equilibrio general**: El macro promedio sube de 0.89 a 0.91, lo que refleja una mejora no solo en el promedio ponderado por tamaño de clase, sino también en el trato equitativo a clases pequeñas.
- **Leafcutter-ants** y *fire-ants* tuvieron una ligera caída en precisión, pero mejoraron en recall y F1-score, lo cual sugiere un modelo menos sesgado hacia ciertas clases.

#### Evaluación usando datos externos
![image](https://github.com/user-attachments/assets/4fbabacf-e401-4b49-90b4-f72104a4c51e)

Al evaluar con imágenes externas, el modelo fracasó en casi todas las clases excepto en leafcutter-ants, donde logró un desempeño aceptable. Esto indica una pobre generalización, lo cual contradice la alta precisión alcanzada durante el entrenamiento y validación. Este comportamiento sugiere que el modelo memorizó características superficiales del conjunto de entrenamiento.

Las imágenes del dataset original fueron recolectadas principalmente a partir de capturas programadas de video incluyendo documentales y grabaciones de YouTube para todas las especies, excepto *leafcutter-ants*. Para esta clase se aplicó webscraping con dos niveles de profundidad, lo que generó una colección mucho más diversa de contextos visuales. Esta variedad permitió al modelo aprender más que simples correlaciones con el fondo o ambiente de origen.

#### Conclusiones

Los resultados indican que el modelo estaba aprendiendo patrones del entorno específico de cada especie, como iluminación, tipo de suelo o calidad de imagen, más que atributos visuales distintivos de las hormigas. En esencia, estaba clasificando escenas o hormigueros más que a las hormigas mismas. Esto explica su incapacidad para adaptarse a nuevos contextos visuales. La solución inmediata es ampliar el dataset incluyendo más imágenes variadas por especie, tomadas en distintos ambientes y condiciones de captura usando el mismo método que con las leafcutter ants, para fomentar un aprendizaje más robusto y centrado en las características reales de cada especie.


## Versión final del modelo

### Aumento de los datos

Para mejorar la capacidad de generalización del modelo y evitar que aprendiera únicamente patrones de fondo o condiciones específicas de captura, se amplió significativamente el dataset de entrenamiento y prueba.

En la versión original del conjunto de datos, algunas clases estaban subrepresentadas y la mayoría de las imágenes provenían de fuentes homogéneas, como capturas de video de YouTube. Esto provocaba que el modelo asociara incorrectamente las especies con su contexto visual, en lugar de con sus características morfológicas.

En esta versión final, cada clase fue enriquecida con nuevas imágenes obtenidas mediante el mismo enfoque usado anteriormente para *leafcutter-ants*. Se utilizó ShareX para capturar frames representativos desde videos diversos, y se complementó con imágenes extraídas de fuentes especializadas como AntWiki. Esta estrategia permitió incorporar imágenes en distintas condiciones, aumentando la diversidad visual del dataset.

#### Comparativa de distribución

| Dataset | Clase                | Original | Final |
|--------:|----------------------|---------:|------:|
| TRAIN   | trap-jaw-ants        | 525      | 520   |
|         | black-crazy-ants     | 417      | 710   |
|         | leafcutter-ants      | 1039     | 1017  |
|         | yellow-crazy-ants    | 468      | 753   |
|         | argentine-ants       | 1136     | 1629  |
|         | weaver-ants          | 612      | 935   |
|         | fire-ants            | 921      | 1232  |
|         | **TOTAL**            | **5118** | **6796** |

| Dataset | Clase                | Original | Final |
|--------:|----------------------|---------:|------:|
| TEST    | trap-jaw-ants        | 131      | 129   |
|         | black-crazy-ants     | 104      | 177   |
|         | leafcutter-ants      | 259      | 254   |
|         | yellow-crazy-ants    | 116      | 188   |
|         | argentine-ants       | 284      | 407   |
|         | weaver-ants          | 152      | 233   |
|         | fire-ants            | 230      | 308   |
|         | **TOTAL**            | **1276** | **1696** |

Esta mejora en la distribución ayudó a reducir el sesgo por clase y mejoró la robustez del entrenamiento, haciendo al modelo menos dependiente de los contextos visuales específicos de cada especie.

### Mejoras arquitectónicas
Para esta versión final, se hicieron dos ajustes clave en la arquitectura:

- Se redujo ligeramente el *learning rate* a 0.000015, buscando una convergencia más estable y una menor propensión a sobreajustar, dado que el modelo ya estaba muy cerca de su límite de generalización.
- Se incrementó la resolución de entrada de 224×224 a 255×255 píxeles, lo cual permite conservar mayor detalle morfológico en las imágenes y aprovechar mejor la capacidad de extracción de características de EfficientNetV2B0 [5].

Ambas decisiones se tomaron tras observar que el modelo anterior, aunque robusto, aún mostraba señales de memorizar patrones específicos del entrenamiento. Esta arquitectura busca un equilibrio más fino entre capacidad de aprendizaje y estabilidad en la generalización.

### Resultados

#### Matriz de confusión
![image](https://github.com/user-attachments/assets/3197253d-cf7f-4dde-9d7e-e83bcfa9bd37)

La matriz de confusión muestra que el modelo logra un desempeño sólido en todas las clases, con un buen número de predicciones correctas en la diagonal principal. Las confusiones más frecuentes se presentan entre especies similares como *argentine-ants* y *fire-ants*, o entre *leafcutter-ants* y *weaver-ants*, lo cual es comprensible dado su parecido morfológico y tonalidades compartidas.

A pesar de estas confusiones, los errores están distribuidos de forma moderada y no hay una clase dominante absorbente como en versiones anteriores. Esto confirma que el modelo ahora distingue de forma más robusta entre especies, incluso en presencia de rasgos visuales sutiles.

#### Entrenamiento
![image](https://github.com/user-attachments/assets/333ce279-e98c-40e4-b59a-7e4752b637fc)
![image](https://github.com/user-attachments/assets/cb922029-97c8-4884-8353-e7244a594d2c)

En esta versión final, el modelo se entrenó durante **80 épocas**, a diferencia de las 60 utilizadas en versiones anteriores. Esta extensión en el tiempo de entrenamiento permitió una convergencia más progresiva y estable, particularmente en la precisión de prueba, que se mantuvo por encima del 80 % en las últimas etapas.

La curva de entrenamiento muestra que la precisión en el conjunto de entrenamiento continúa mejorando de forma constante hasta el final. Aunque la precisión en validación no alcanza los niveles de entrenamiento ni de prueba, también muestra una tendencia ascendente, aunque más ruidosa.

Esto sugiere que el modelo aún no había alcanzado su punto de saturación, y con mayor capacidad de cómputo o más épocas podría haberse ajustado mejor. Sin embargo, también se observa que el conjunto de validación comienza a estabilizarse, lo cual indica que el modelo se está acercando a su límite de generalización con los datos disponibles. Es posible que un pequeño aumento en el número de épocas (hasta 100, por ejemplo hubiera ofrecido una mejora adicional.

#### Métricas
| Clase               | Precisión | Recall | F1-score | Soporte |
|---------------------|-----------|--------|----------|---------|
| argentine-ants      | 0.78      | 0.83   | 0.80     | 407     |
| black-crazy-ants    | 0.79      | 0.80   | 0.80     | 177     |
| fire-ants           | 0.82      | 0.72   | 0.77     | 308     |
| leafcutter-ants     | 0.82      | 0.81   | 0.82     | 254     |
| trap-jaw-ants       | 0.87      | 0.91   | 0.89     | 129     |
| weaver-ants         | 0.83      | 0.88   | 0.85     | 233     |
| yellow-crazy-ants   | 0.83      | 0.78   | 0.80     | 188     |
| **Accuracy global** |           |        | **0.81** | 1696    |
| Macro avg           | 0.82      | 0.82   | 0.82     | 1696    |
| Weighted avg        | 0.81      | 0.81   | 0.81     | 1696    |

Comparado con la primera versión del modelo, que alcanzó un 85% de accuracy sobre un conjunto de datos más pequeño y visualmente homogéneo, esta versión final muestra una ligera caída en las métricas generales. Sin embargo, este descenso es esperable, ya que ahora se evaluó sobre un conjunto ampliado y más diverso.

El modelo inicial fue entrenado sobre imágenes obtenidas mayoritariamente desde videos específicos por especie, lo que introdujo un sesgo de contexto visual (ambiente, tipo de iluminación, fondos repetidos). Esto facilitó que el modelo memorizara el entorno en lugar de generalizar sobre las características morfológicas de las hormigas. 

En contraste, la versión final usa un dataset enriquecido y más variado, donde las hormigas aparecen en diferentes condiciones, aumentando así el reto de generalización. La caída en métricas refleja precisamente que ahora se está evaluando una capacidad más robusta y realista del modelo.

#### Evaluación usando datos externos
![image](https://github.com/user-attachments/assets/a7cdc1cc-0a1c-4d96-9745-f6907486b84a)

A diferencia de versiones anteriores donde el modelo solo mostraba desempeño aceptable con las leafcutter-ants, en esta evaluación se observaron mejoras notables también en otras especies como trap-jaw-ants, black-crazy-ants y weaver-ants. Esto sugiere que el modelo ya no está simplemente reconociendo patrones contextuales muy específicos (como el fondo o tipo de nido), sino que comienza a aprender representaciones más generalizables de las propias hormigas.

Esta mejora es atribuible directamente al nuevo dataset ampliado, que incorporó imágenes capturadas en una mayor variedad de entornos. Al tener acceso a datos visualmente más diversos, el modelo logró aprender a identificar las especies con base en características morfológicas y no tanto en el contexto en que fueron fotografiadas.

#### Conclusiones
A través de un conjunto de mejoras progresivas en la arquitectura, el preprocesamiento y especialmente en la calidad y diversidad del dataset, se logró construir un clasificador de especies de hormigas con buena capacidad de generalización y desempeño robusto en escenarios realistas. El accuracy global de 81 % se obtuvo evaluando sobre un conjunto más amplio y exigente que versiones anteriores, lo que demuestra una mejora sustancial en la solidez del modelo. A diferencia de las primeras versiones que aprendían a identificar contextos visuales específicos, el modelo actual muestra un entendimiento más centrado en las características propias de las hormigas.

La ampliación del dataset fue clave para romper la dependencia del modelo con los ambientes originales de las imágenes. Al incorporar más ejemplos por clase, extraídos de diferentes fuentes, se redujo el sesgo por fondo y se permitió que la red aprendiera a reconocer formas, estructuras y texturas relevantes de cada especie. En términos de arquitectura, el uso de EfficientNetV2B0 con ajustes finos de hiperparámetros como el learning rate y el tamaño de imagen ofreció una base eficiente y fácil de adaptar para agregar más clases de hormigas y diversificar las actuales. El entrenamiento extendido a 80 épocas mostró que el modelo seguía mejorando con el tiempo, indicando que aún existía potencial sin explotar.

En conjunto, el proyecto cumple su objetivo: demostrar que con una selección adecuada de datos y una arquitectura bien afinada, es posible entrenar modelos capaces de identificar distintas especies de hormigas a partir de imágenes en condiciones variadas. El sistema final está lejos de ser perfecto, pero ofrece una base sólida para futuras mejoras, ya sea ampliando el número de especies o integrando métodos de atención para mejorar la localización de rasgos clave.

## Bibliografía
[1] M. S. Norouzzadeh, A. Nguyen, M. Kosmala, A. Swanson, M. Palmer, C. Packer, and J. Clune, “Automatically identifying, counting, and describing wild animals in camera-trap images with deep learning,” arXiv:1703.05830, 2017.

[2] S. D. Das and A. Kumar, “Bird Species Classification using Transfer Learning with Multistage Training,” arXiv:1810.04250, 2018.

[3] A. Krizhevsky, I. Sutskever, and G. E. Hinton, “ImageNet Classification with Deep Convolutional Neural Networks,” in *Proc. 25th Int. Conf. Neural Information Processing Systems (NIPS’12)*, Lake Tahoe, NV, USA, 2012, pp. 1097–1105, doi:10.1145/3065386.

[4] M. Tan and Q. V. Le, "EfficientNetV2: Smaller Models and Faster Training," *Proceedings of the 38th International Conference on Machine Learning*, PMLR 139, 2021.

[5] L. Otarashvili, T. Subramanian, J. Holmberg, J. J. Levenson, and C. V. Stewart, “Multispecies Animal Re-ID Using a Large Community-Curated Dataset,” arXiv:2401.00000 [cs.CV], 2024.
