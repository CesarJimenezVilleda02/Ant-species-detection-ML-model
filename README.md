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

Se añadió una nueva clase: **leafcutter-ants**, debido a su relevancia como plaga en cultivos de la región de Huichapan, Hidalgo. Las imágenes fueron obtenidas manualmente utilizando la herramienta ShareX para capturar frames de diversos videos en YouTube que mostraban estas hormigas en diferentes contextos (cautiverio, nidos, campo abierto). También se recolectaron imágenes desde la plataforma especializada AntWiki.

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

## Bibliografía
[1] M. S. Norouzzadeh, A. Nguyen, M. Kosmala, A. Swanson, M. Palmer, C. Packer, and J. Clune, “Automatically identifying, counting, and describing wild animals in camera-trap images with deep learning,” arXiv:1703.05830, 2017.

[2] S. D. Das and A. Kumar, “Bird Species Classification using Transfer Learning with Multistage Training,” arXiv:1810.04250, 2018.

[3] A. Krizhevsky, I. Sutskever, and G. E. Hinton, “ImageNet Classification with Deep Convolutional Neural Networks,” in *Proc. 25th Int. Conf. Neural Information Processing Systems (NIPS’12)*, Lake Tahoe, NV, USA, 2012, pp. 1097–1105, doi:10.1145/3065386.

[4] R. Hadipour-Roknia, E. Askari Asli-Ardeh, A. Jahanbakhshi, I. Esmaili Paeen-Afrakotic, and S. Sabzi, “Intelligent detection of citrus fruit pests using machine vision system and convolutional neural network through transfer learning technique,” *Computers in Biology and Medicine*, vol. 155, p. 106611, Feb. 2023, doi:10.1016/j.compbiomed.2023.106611.

