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

### `fill_mode = "wrap"`
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



