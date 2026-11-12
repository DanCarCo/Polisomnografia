# Bitácora de decisiones del equipo — Polisomnografía (HMM)

**Equipo:** Daniel Carvajal Correa, Jhon Edison Sandoval Barreto y Jhon Edison Velásquez Giraldo · **Grupo:** PREICA2602B020045
**Asignatura:** Simulación · Institución Universitaria Digital de Antioquia · **Docente:** Julián Andrés Loaiza López
**Repositorio:** https://github.com/DanCarCo/Polisomnografia

## Cómo leer esta bitácora

El formato es el del laboratorio «Investiga con IA. Decide con criterio» (`Investiga_con_IA_Decide_con_criterio.html`): cada registro tiene fecha, herramienta, consulta, propuesta, verificación, decisión y justificación. La decisión es una de **Pendiente, Aceptar, Modificar o Rechazar**. A esos campos se añadió **Decide** (quién tomó la decisión) y, cuando existieron, **Alternativas descartadas**.

Esta bitácora la **reconstruyó Claude Code a partir de la conversación de trabajo y del estado del repositorio**; no la escribió el equipo. Cada registro indica quién decidió:

- **El usuario (chat):** decidió expresamente en la conversación; sus palabras figuran en la justificación.
- **Claude Code (autonomía delegada):** decidió la IA. El usuario le delegó la autonomía de decisión: al inicio escribió «Tienes autorización para crear las carpetas y los archivos y los códigos en esta ruta», y el 2026-11-09 confirmó «Yo te di autonomía en las decisiones». Estas decisiones se adoptaron **sin una revisión individual del equipo** y el equipo puede revertirlas.
Lo que hizo el equipo por su cuenta (verificar las fuentes originales, elegir APA para el Word, completar los enlaces y hacer el video) consta **según lo declarado por el usuario**; la IA no lo comprobó (Registros 7, 23, 24, 26, 27, 28 y 30).

Si el equipo ya registró decisiones en el HTML del laboratorio, esas quedan guardadas en el navegador donde se usó (no en este repositorio): expórtenlas desde «Abrir bitácora y exportar» (Informe Markdown o Respaldo JSON) y únanlas a este archivo.

## Bitácora de decisiones

### Registro 1 · Tema y alcance del trabajo

**Fecha:** 2026-11-08

**Herramienta:** Claude Code (Anthropic); modelo de la sesión actual: Claude Sonnet 5.5

**Consulta:** El usuario pidió hacer todo lo que pide `Instrucciones.pdf` con el tema «Análisis de Etapas del Sueño (Polisomnografía)», tomando los demás archivos como material complementario, y autorizó crear carpetas, archivos y código en `D:\Codigos\Polisomnografia`.

**Propuesta:** Un HMM de cuatro estados (Despierto, Sueño Ligero, Sueño Profundo, REM) y cuatro símbolos (ondas alfa, theta, delta y movimientos oculares rápidos), con época de un minuto y noche de 480 épocas. Se usan las matrices π, A y B del Ejemplo 2 del enunciado. Es un estudio de simulación, sin registros de pacientes.

**Verificación:** Las pruebas automáticas (36; todas pasaban hasta el cambio de nombres de las figuras, Registro 28) comprueban que cada fila de A y de B suma 1 y que A[REM → Profundo] = 0. El artículo declara que no hay validez clínica y que las matrices no se estimaron de datos reales.

**Decide:** El usuario (chat)

**Decisión:** Aceptar

**Justificación:** El tema lo eligió el usuario («el tema que vamos a usar es…»). Las matrices vienen del enunciado del propio curso, no son valores propuestos por la IA.

### Registro 2 · Escenarios E1–E5 y barridos de sensibilidad

**Fecha:** 2026-11-08

**Herramienta:** Claude Code (Anthropic); modelo de la sesión actual: Claude Sonnet 5.5

**Consulta:** Cómo cumplir el requisito de al menos tres escenarios con media, desviación típica, mínimo, máximo e IC95 %.

**Propuesta:** Cinco escenarios de 100 noches de 480 épocas: E1 nominal; E2 transición perturbada (cada fila de A ~ Dirichlet(20·aᵢ)); E3 emisión degradada (B' = 0.7·B + 0.3·U); E4 emisión continua (Gauss × Poisson); E5 sueño consolidado (aᵢᵢ = 0.80, 0.95, 0.95, 0.95). Además, barridos de A, B y persistencia, estudio de la condición inicial y 20 semillas maestras.

**Verificación:** `results/escenarios_resumen.csv`. Exactitud de Viterbi: E1 71.4 % (IC95 70.8–71.9), E5 92.7 %. Los parámetros de E2 a E5 los eligió la IA; no vienen del enunciado.

**Decide:** Claude Code (autonomía delegada)

**Decisión:** Aceptar

**Justificación:** Cubre el requisito de tres escenarios o más con un diseño que separa dos efectos: la pérdida de información (emisión) y el desajuste del modelo (transición). Los valores concretos son elección de la IA y el equipo puede cambiarlos.

### Registro 3 · Semillas y reproducibilidad

**Fecha:** 2026-11-08

**Herramienta:** Claude Code (Anthropic); modelo de la sesión actual: Claude Sonnet 5.5

**Consulta:** Cómo garantizar réplicas independientes y resultados reproducibles, sin repetir la misma semilla en cada réplica.

**Propuesta:** Semilla maestra 42 y `numpy.random.SeedSequence.spawn` para derivar una semilla hija independiente por réplica y por experimento; versiones exactas en `requirements.txt` y `results/environment.json`; sumas SHA-256 de los resultados en `results/SHA256SUMS.txt`.

**Verificación:** Dos ejecuciones independientes dieron sumas SHA-256 idénticas. `sha256sum -c` confirma los 26 archivos (comprobado el 2026-11-09 y de nuevo el 2026-11-12). La media entre 20 semillas maestras varía con DE 0.0029, frente a un error estándar teórico de 0.0026. Con `.gitattributes` (LF) la comprobación sigue siendo válida en un clon hecho en Windows.

**Decide:** Claude Code (autonomía delegada)

**Decisión:** Aceptar

**Justificación:** El laboratorio exige registrar versiones, parámetros y semillas y no repetir la misma semilla. La igualdad bit a bit solo se comprobó en este entorno; con otras versiones de NumPy, SciPy o sistema operativo pueden cambiar los últimos dígitos.

### Registro 4 · Intervalos de confianza y pruebas estadísticas

**Fecha:** 2026-11-08

**Herramienta:** Claude Code (Anthropic); modelo de la sesión actual: Claude Sonnet 5.5

**Consulta:** Qué intervalo de confianza usar para la media de la exactitud con 100 réplicas y cómo comparar escenarios.

**Propuesta:** IC95 % con la t de Student (n = 100, t = 1.984) y contraste con bootstrap percentil. Comparaciones con la prueba t de Welch, Mann-Whitney y el d de Cohen, con corrección de Holm.

**Alternativas descartadas:** Usar el valor normal 1.96, que subestima el intervalo con 100 réplicas.

**Verificación:** Implementación en `src/utils_stats.py`, cubierta por las pruebas automáticas. Resultados en `results/escenarios_comparaciones.csv`.

**Decide:** Claude Code (autonomía delegada)

**Decisión:** Aceptar

**Justificación:** Las réplicas son independientes y su varianza se estima de la muestra, lo que corresponde a la t de Student; el bootstrap sirve de contraste.

### Registro 5 · Viterbi en log-espacio en lugar del lineal de la plantilla

**Fecha:** 2026-11-08

**Herramienta:** Claude Code (Anthropic); modelo de la sesión actual: Claude Sonnet 5.5

**Consulta:** El Viterbi de la plantilla del curso trabaja en dominio lineal. ¿Es adecuado para noches de 480 épocas?

**Propuesta:** La plantilla del curso trae Viterbi en dominio lineal. Cambio adoptado: Forward-Backward escalado y Viterbi en log-espacio; el lineal se conserva solo para mostrar su *underflow*.

**Alternativas descartadas:** Mantener el Viterbi lineal de la plantilla como método principal.

**Verificación:** Forward, Viterbi y posteriores coinciden con la enumeración exhaustiva de todas las trayectorias y con `hmmlearn`. El Viterbi lineal hace *underflow* en el 84 % de las noches de 480 épocas y en el 100 % con 600 o más (`results/estabilidad_numerica.csv`).

**Decide:** Claude Code (autonomía delegada)

**Decisión:** Modificar

**Justificación:** Se cambió el método de la plantilla porque el Viterbi lineal falla con las longitudes de este estudio; se adoptó el de log-espacio.

### Registro 6 · Referencias bibliográficas: elección y descarte

**Fecha:** 2026-11-08

**Herramienta:** Claude Code (Anthropic); modelo de la sesión actual: Claude Sonnet 5.5, con búsqueda web y consulta a Crossref

**Consulta:** Reunir al menos 10 referencias de los últimos cinco años en formato IEEE y revisar las que trae la plantilla del curso.

**Propuesta:** 29 referencias (18 recientes, 8 clásicas y 3 de software). Se descartan las cinco referencias de la plantilla (Smith y Johnson, García y López, Chen, Müller y Silva, Zhao) y las cifras que las acompañan.

**Alternativas descartadas:** Reutilizar las cinco referencias de la plantilla del curso.

**Verificación:** Los metadatos de cada referencia se comprobaron contra Crossref por su DOI (o contra la página del editor cuando no hay DOI). Las cinco de la plantilla no existen en Crossref. **No se verificó** la indexación en Scopus, Web of Science o IEEE.

**Decide:** Claude Code (autonomía delegada)

**Decisión:** Aceptar

**Justificación:** Cumple el requisito de al menos 10 referencias recientes y evita citar trabajos que no existen. La lectura de las fuentes originales la hizo el equipo (Registro 7).

### Registro 7 · Verificación de las fuentes originales

**Fecha:** 2026-11-09

**Herramienta:** Claude Code (Anthropic); modelo de la sesión actual: Claude Sonnet 5.5

**Consulta:** ¿Quién comprueba que cada afirmación del estado del arte coincide con lo que dice el artículo original?

**Propuesta:** Que el equipo abra cada fuente original y verifique las citas y las cifras de la tabla del estado del arte (por ejemplo, la exactitud de 83.98 % de Yang et al.).

**Verificación:** El usuario informó el 2026-11-09 que el equipo ya verificó las fuentes originales de las referencias y las citas. La IA no presenció ni comprobó esa verificación; por su parte solo había leído resúmenes y metadatos, no los artículos completos.

**Decide:** El usuario (chat)

**Decisión:** Aceptar

**Justificación:** Declaración del usuario: el equipo ya verificó las fuentes originales de las referencias y las citas. El laboratorio reserva esa verificación al estudiante.

### Registro 8 · Alcance de la revisión de literatura

**Fecha:** 2026-11-08

**Herramienta:** Claude Code (Anthropic); modelo de la sesión actual: Claude Sonnet 5.5

**Consulta:** Cómo presentar la búsqueda bibliográfica y el vacío de investigación sin afirmar más de lo que se comprobó.

**Propuesta:** Describir la revisión como «exploratoria y dirigida, no sistemática» (Crossref, OpenAlex, Semantic Scholar y web) y limitar la afirmación del vacío a esa búsqueda: no se afirma que sea el único ni el primer trabajo.

**Alternativas descartadas:** Presentarla como revisión sistemática; mantener expresiones como «único escenario» o «igual al error estándar», que sobreafirmaban.

**Verificación:** El texto está en las secciones «Estado del arte» y «Limitaciones» del informe.

**Decide:** Claude Code (autonomía delegada)

**Decisión:** Aceptar

**Justificación:** Es lo que respalda la búsqueda realizada; el laboratorio pide no llamar sistemática a una búsqueda exploratoria.

### Registro 9 · Baum-Welch: inicialización y resultados desfavorables

**Fecha:** 2026-11-08

**Herramienta:** Claude Code (Anthropic); modelo de la sesión actual: Claude Sonnet 5.5

**Consulta:** Cómo evaluar el aprendizaje de parámetros con Baum-Welch sin mezclar datos de entrenamiento y de prueba.

**Propuesta:** Entrenar con 1, 5 y 20 noches, con tres reinicios y dos inicializaciones (aleatoria e informada). Noches de prueba disjuntas de las de entrenamiento (partición por noche). Se conservan y reportan los resultados desfavorables.

**Verificación:** `results/aprendizaje_resumen.csv`. Con 20 noches, la inicialización aleatoria llega a 62.1 % de exactitud en prueba (mínimo 42.5 %), la informada a 69.7 % y el modelo verdadero a 71.1 %. Ninguna variante supera con claridad al clasificador sin memoria (70.1 %).

**Decide:** Claude Code (autonomía delegada)

**Decisión:** Aceptar

**Justificación:** El laboratorio pide separar entrenamiento y prueba y conservar los resultados negativos. Se reportó la debilidad de la inicialización aleatoria en lugar de omitirla.

### Registro 10 · Alteraciones simuladas y su detección

**Fecha:** 2026-11-08

**Herramienta:** Claude Code (Anthropic); modelo de la sesión actual: Claude Sonnet 5.5

**Consulta:** Cómo cumplir el requisito de simular alteraciones del sueño y detectarlas.

**Propuesta:** Dos alteraciones: *sueño fragmentado* (se suma una masa δ = 0.15 a la transición hacia Despierto desde las etapas de sueño) y *ausencia de REM* (se anulan las transiciones hacia REM, o se reducen según la severidad del barrido, conservando la fila de REM como distribución válida). Cada noche se clasifica por máxima log-verosimilitud Forward; hay alertas por puntuación z frente a 300 noches de referencia y poder de detección con AUC e IC95 % bootstrap (500 remuestreos), con controles nulos.

**Alternativas descartadas:** Poner la fila de REM en ceros para «quitar» el REM: dejaba una matriz de transición inválida (Registro 13).

**Verificación:** `results/patologias_*.csv`. Sensibilidad de 0.99 para sueño fragmentado y 1.00 para ausencia de REM; los controles nulos dan un AUC de 0.500 y 0.536, compatibles con el azar.

**Decide:** Claude Code (autonomía delegada)

**Decisión:** Aceptar

**Justificación:** Cubre el requisito con un diseño que incluye controles nulos. Los informes clínicos son educativos y no constituyen diagnóstico.

### Registro 11 · Señales sintéticas y definición operativa de los símbolos

**Fecha:** 2026-11-08

**Herramienta:** Claude Code (Anthropic); modelo de la sesión actual: Claude Sonnet 5.5

**Consulta:** El enunciado pide señales de EEG simuladas y un hipnograma; ¿cómo relacionar la señal con los símbolos del HMM?

**Propuesta:** Simular por época una señal de EEG/EOG dominada por la banda del símbolo observado y recuperar el símbolo con una regla operativa (frecuencia pico de la densidad espectral de Welch y conteo de ráfagas en el EOG).

**Verificación:** `results/consistencia_extraccion_senales.csv`. La regla acierta el 100 %, lo cual es esperable porque la señal se genera desde el propio símbolo.

**Decide:** Claude Code (autonomía delegada)

**Decisión:** Aceptar

**Justificación:** Se presenta solo como **verificación de consistencia interna**: no valida la fisiología y el informe lo dice así.

### Registro 12 · Decodificadores, línea base y «techo»

**Fecha:** 2026-11-08

**Herramienta:** Claude Code (Anthropic); modelo de la sesión actual: Claude Sonnet 5.5

**Consulta:** Con qué comparar el rendimiento de Viterbi para saber cuánto aporta el contexto temporal.

**Propuesta:** Comparar cuatro decodificadores por noche: Viterbi; posterior (Forward-Backward); un clasificador de Bayes sin memoria temporal (usa B y la distribución estacionaria, pero ignora A) como línea base sin HMM; y el techo, definido como el decodificador posterior con el modelo verdadero.

**Alternativas descartadas:** Definir el techo como Viterbi con el modelo verdadero: salía peor que un decodificador con el modelo desajustado, lo que no tiene sentido para un techo.

**Verificación:** `results/escenarios_decodificadores_pareados.csv`. En E1 el posterior alcanza 72.8 %, Viterbi 71.4 % y la línea base 70.1 %.

**Decide:** Claude Code (autonomía delegada)

**Decisión:** Modificar

**Justificación:** Se corrigió la definición del techo durante el desarrollo, tras detectar la incoherencia, y se documentó en el informe.

### Registro 13 · Correcciones técnicas durante el desarrollo

**Fecha:** 2026-11-08

**Herramienta:** Claude Code (Anthropic); modelo de la sesión actual: Claude Sonnet 5.5

**Consulta:** Fallos detectados al probar el código y los resultados.

**Propuesta:** (i) `no_rem_hmm` dejaba la fila de REM en ceros: se reescribió para que la fila siga siendo una distribución válida. (ii) Las semillas del aprendizaje con un `spawn` con estado salían inconsistentes: se pasó a una semilla determinista por etiqueta. (iii) Los saltos de línea CRLF de Windows rompían `sha256sum -c`: se fuerza LF al escribir. (iv) Algunas cifras del README estaban mal redondeadas: se corrigieron desde las macros generadas.

**Verificación:** 36 pruebas pasaban (una falla desde el cambio de nombres de las figuras, Registro 28); `sha256sum -c` da 26 archivos OK (reverificado el 2026-11-12); dos ejecuciones independientes producen los mismos resultados.

**Decide:** Claude Code (autonomía delegada)

**Decisión:** Modificar

**Justificación:** Cada corrección elimina un error o una inconsistencia comprobable.

### Registro 14 · Cifras generadas por código y pruebas de consistencia

**Fecha:** 2026-11-08

**Herramienta:** Claude Code (Anthropic); modelo de la sesión actual: Claude Sonnet 5.5

**Consulta:** Cómo evitar que las cifras del artículo se desvíen de los resultados.

**Propuesta:** No escribir cifras a mano: `src/latex_export.py` genera macros y tablas de LaTeX desde `results/*.csv`, y `tests/test_paper_consistency.py` valida citas, etiquetas y macros. El informe Word se genera desde la misma fuente.

**Alternativas descartadas:** Copiar las cifras manualmente al texto.

**Verificación:** Las pruebas de consistencia pasaban y las cifras del informe coinciden con las de `results/`. Desde el cambio de nombres de las figuras falla una prueba que comprueba que existan los archivos de figuras que cita `main.tex` (Registro 28).

**Decide:** Claude Code (autonomía delegada)

**Decisión:** Aceptar

**Justificación:** Cumple el acuerdo de no inventar resultados y facilita rehacer el informe si cambian los datos.

### Registro 15 · Estructura, formato y extensión del artículo

**Fecha:** 2026-11-08

**Herramienta:** Claude Code (Anthropic); modelo de la sesión actual: Claude Sonnet 5.5

**Consulta:** Cómo cumplir el formato de `Instrucciones.pdf`: artículo en español, tamaño carta, interlineado 1.5, entre 10 y 15 páginas sin referencias ni anexos, citas IEEE.

**Propuesta:** Artículo en LaTeX, carta, 11 pt, interlineado 1.5 y márgenes de 2.3 cm, con las secciones pedidas. Las tablas y figuras secundarias van a los anexos; la figura de matrices de confusión pasó al Anexo A para ganar margen.

**Alternativas descartadas:** Dejar todas las figuras y tablas en el cuerpo: la estimación inicial era de 16.7 páginas.

**Verificación:** Estimación de 13.7 páginas tras reorganizar. Compilado con Tectonic (Registro 17): cuerpo de unas 14.4 páginas, dentro del límite de 15.

**Decide:** Claude Code (autonomía delegada)

**Decisión:** Aceptar

**Justificación:** Cumple el límite de páginas. El número exacto puede variar unas líneas en Overleaf con pdfLaTeX.

### Registro 16 · Figuras: generación por código y paleta de colores

**Fecha:** 2026-11-08

**Herramienta:** Claude Code (Anthropic); modelo de la sesión actual: Claude Sonnet 5.5, con el skill de visualización de datos

**Consulta:** Cómo producir figuras de calidad publicable (300 dpi) que se distingan bien, también para personas con daltonismo.

**Propuesta:** 12 figuras PNG a 300 dpi generadas por código (`src/plotting.py`) con una paleta fija por etapa: Despierto #2a78d6, Ligero #008300, Profundo #8f2fa3 y REM #e34948.

**Verificación:** La paleta se validó con el script del skill de visualización (todas las parejas pasan). Las figuras se revisaron visualmente y se corrigieron solapes de etiquetas y bucles recortados.

**Decide:** Claude Code (autonomía delegada)

**Decisión:** Aceptar

**Justificación:** Las figuras se regeneran con `python main_simulation.py`; no se dibujaron ni retocaron a mano.

### Registro 17 · Compilador LaTeX para generar y revisar el PDF

**Fecha:** 2026-11-08

**Herramienta:** Claude Code (Anthropic) con Tectonic 0.17

**Consulta:** ¿Instalo un compilador LaTeX para generar y revisar el PDF? (el equipo no tenía ninguno instalado).

**Propuesta:** Descargar Tectonic 0.17 (release oficial en GitHub, 21 MB) a una carpeta temporal, ejecutarlo desde fuera de ella y compilar `paper/main.tex`.

**Verificación:** El sistema bloqueó el primer intento. Tras el permiso explícito del usuario se descargó el archivo, cuyo tamaño coincide con el del release (21 060 223 bytes). El artículo compiló sin avisos de TeX: 23 páginas, con el cuerpo en unas 14.4 (el límite es 15, sin contar referencias ni anexos).

**Decide:** El usuario (chat)

**Decisión:** Aceptar

**Justificación:** Respuestas del usuario: «Sí, instalar» y, después del bloqueo, «Reintenta con permiso».

### Registro 18 · Creación del repositorio de GitHub

**Fecha:** 2026-11-08

**Herramienta:** Claude Code (Anthropic); modelo de la sesión actual: Claude Sonnet 5.5

**Consulta:** El usuario pidió un `.gitignore` para que en GitHub solo quede lo estrictamente necesario y ordenó: «Aún no crees el repositorio en github».

**Propuesta:** Crear el `.gitignore`; no ejecutar `git init` ni crear el repositorio hasta que el usuario lo pidiera, y entonces explicarle los comandos.

**Verificación:** La IA no creó el repositorio. El usuario lo creó por su cuenta: commit `ffe0964` del 2026-11-08 y push a https://github.com/DanCarCo/Polisomnografia (público). El repositorio rastrea 73 archivos: resultados, figuras y tablas, código, pruebas y las diapositivas del pitch.

**Decide:** El usuario (chat)

**Decisión:** Aceptar

**Justificación:** Instrucción explícita del usuario («Aún no crees el repositorio en github»).

### Registro 19 · Qué se excluye de GitHub (.gitignore)

**Fecha:** 2026-11-08

**Herramienta:** Claude Code (Anthropic); modelo de la sesión actual: Claude Sonnet 5.5

**Consulta:** El usuario pidió agregar al `.gitignore` los archivos `.gitignore`, `main.pdf` y `main.tex`, y después `video/guion_pitch.md`.

**Propuesta base de la IA:** Excluir entornos virtuales, cachés, auxiliares de LaTeX, videos, el material del curso (`Instrucciones.pdf`, `articulo.pdf`, `OVERLEAF code.docx`), `.claude/` y archivos de editores o sistema.

**Verificación:** Se probó con un repositorio temporal y se confirmó con `git check-ignore` que `.gitignore`, `main.tex`, `main.pdf` y `video/guion_pitch.md` quedan fuera.

**Decide:** El usuario (chat)

**Decisión:** Modificar

**Justificación:** El usuario añadió esas cuatro exclusiones a la propuesta. No explicó el motivo; su objetivo declarado fue que en GitHub solo quede lo estrictamente necesario. Consecuencia: el artículo en LaTeX, su PDF y el guion del pitch no se suben, y quien clone el repositorio no tendrá las reglas de `.gitignore`.

### Registro 20 · Finales de línea (.gitattributes)

**Fecha:** 2026-11-08

**Herramienta:** Claude Code (Anthropic); modelo de la sesión actual: Claude Sonnet 5.5

**Consulta:** Al hacer `git add .`, Git avisó «LF will be replaced by CRLF» (`core.autocrlf=true` en Windows).

**Propuesta:** Agregar `.gitattributes` con `* text=auto eol=lf` y los PNG, PDF y PPTX como binarios, para que quien clone en Windows reciba los archivos con LF.

**Verificación:** Las sumas SHA-256 de `results/` se calcularon con LF; con CRLF, `sha256sum -c` fallaría en un clon. El archivo está versionado (aparece en `git ls-files`) y no está ignorado.

**Decide:** Claude Code (autonomía delegada) y el usuario

**Decisión:** Aceptar

**Justificación:** La IA creó el archivo y el usuario lo adoptó al incluirlo en su commit.

### Registro 21 · README: contenido y licencia

**Fecha:** 2026-11-08

**Herramienta:** Claude Code (Anthropic); modelo de la sesión actual: Claude Sonnet 5.5

**Consulta:** El usuario pidió ajustar el README: quitar la mención del guion, los autores, Google Colab/Jupyter y toda la sección «Artículo y video», y declarar que no tiene licencia.

**Propuesta inicial de la IA:** README con resultados, estructura, instalación, reproducibilidad, sección de artículo y video, autores y licencia «por definir», más un ejemplo para Google Colab.

**Verificación:** Se aplicaron los cambios y una búsqueda confirmó que no quedan menciones de guion, Colab, Jupyter ni autores. El README sigue citando «el artículo» en tres sitios («Modelo», «Cumplimiento de la actividad» y «Limitaciones»).

**Decide:** El usuario (chat)

**Decisión:** Modificar

**Justificación:** El usuario ordenó declarar que el proyecto «no tiene licencia para uso o distribución» y que «solo se crea para presentar una actividad en la asignatura Simulación de la Institución universitaria digital de Antioquia». Se dejó esa nota en la sección «Licencia».

### Registro 22 · Interpretación de la petición del informe Word

**Fecha:** 2026-11-09

**Herramienta:** Claude Code (Anthropic); modelo de la sesión actual: Claude Sonnet 5.5, con Microsoft Word en modo de solo lectura para exportar y medir páginas

**Consulta:** El usuario escribió: «Agregue un archivo con nombre Carvajal_Sandoval_Velasquez_Polisomnografia, coloca el informe en este archivo, manteniendo el formato que ya tiene dicho archivo».

**Propuesta:** Primera interpretación (equivocada): crear un PDF con ese nombre copiando `main.pdf` a `paper/`.

**Alternativas descartadas:** Esa copia en PDF.

**Verificación:** Al revisar `git status` apareció `Carvajal_Sandoval_Velasquez_Polisomnografia.docx`, que el usuario ya había agregado a la raíz del repositorio: la petición era volcar el informe en ese archivo conservando su formato. Se eliminó la copia en PDF (idéntica a `main.pdf`), se guardó un respaldo del `.docx` original fuera del repositorio y solo se sobrescribió después de comprobar con `cmp` que el archivo no había cambiado desde el respaldo. El sistema bloqueó cerrar a la fuerza un proceso de Word; se cerró de forma ordenada.

**Decide:** Claude Code (autonomía delegada)

**Decisión:** Modificar

**Justificación:** Corrección de un error de interpretación de la IA, detectado antes de modificar el archivo del usuario.

### Registro 23 · Informe en la plantilla Word institucional

**Fecha:** 2026-11-09

**Herramienta:** Claude Code (Anthropic); modelo de la sesión actual: Claude Sonnet 5.5, con Microsoft Word para exportar y revisar el resultado

**Consulta:** Volcar el informe en `Carvajal_Sandoval_Velasquez_Polisomnografia.docx` manteniendo el formato del archivo; el usuario indicó que este archivo sí irá a GitHub.

**Propuesta:** Volcar el contenido del artículo en la plantilla, conservando portada, encabezado, índice y estilos. Resultado de 30 páginas con 235 ecuaciones editables, 24 tablas y 12 figuras. Decisiones de la IA al hacerlo: citas y referencias en IEEE numérico (no APA); omitir el Anexo C (guion del pitch, que el usuario excluyó de GitHub); reemplazar las dos referencias APA de ejemplo de la plantilla; quitar un párrafo vacío de la portada para que la fecha no pasara a la página 2; añadir el enlace del repositorio. Los enlaces de OneDrive y video quedan vacíos.

**Alternativas descartadas:** Convertir las citas a APA, el estilo de la plantilla: `Instrucciones.pdf` pide IEEE.

**Verificación:** Word abrió el archivo sin errores y lo exportó a PDF; se revisó visualmente el PDF exportado y los números del índice quedaron estables en dos pasadas. La versión que generó la IA tenía citas IEEE numéricas y los enlaces de OneDrive y video vacíos. Los cambios posteriores del equipo constan por declaración del usuario (2026-11-09); la IA no comprobó la versión final del archivo.

**Decide:** Claude Code (autonomía delegada) para el formato; el usuario (chat) para el estilo de citas y los enlaces

**Decisión:** Modificar

**Justificación:** El equipo decidió que en el Word las citas quedan en formato APA y completó los enlaces, salvo el del video (declaración del usuario). APA es el estilo de la plantilla institucional; el artículo en LaTeX sigue en IEEE, como pide `Instrucciones.pdf`. El equipo elabora la versión final en un documento externo y la cargará en el repositorio en lugar de la que generó la IA, que se eliminará (declaración del usuario, 2026-11-09). La versión del equipo se cargó el 2026-11-12 (Registro 27).

### Registro 24 · Video pitch

**Fecha:** 2026-11-09

**Herramienta:** Claude Code (Anthropic); modelo de la sesión actual: Claude Sonnet 5.5

**Consulta:** Requisito de `Instrucciones.pdf` que no puede cumplir la IA: un video de 2 a 3 minutos (MP4, 1080p).

**Propuesta:** La IA decidió preparar los materiales: un guion de unas 359 palabras (unos 2.4 minutos) y 8 diapositivas en 16:9 con notas del orador (`video/pitch_polisomnografia.pptx`). Al equipo le corresponde grabar el video.

**Verificación:** El usuario informó el 2026-11-09 que el equipo ya hizo el video y colocó el enlace; el 2026-11-12 aclaró que el enlace del video todavía no está en el documento Word y pidió no tratar ese faltante como un problema. La IA no vio el video. Después informó que ya agregó el enlace del video al Word (Registro 30); la IA no lo comprobó. Las diapositivas que preparó la IA solo se revisaron en su geometría; no se renderizaron. El guion quedó fuera de GitHub por decisión del usuario (Registro 19).

**Decide:** El usuario (chat)

**Decisión:** Aceptar

**Justificación:** Declaración del usuario: el equipo ya hizo el video y agregó su enlace al Word (el enlace faltaba hasta el 2026-11-12; Registro 30).

### Registro 25 · Atribución de las decisiones en esta bitácora

**Fecha:** 2026-11-09

**Herramienta:** Claude Code (Anthropic); modelo de la sesión actual: Claude Sonnet 5.5

**Consulta:** El usuario dijo: «Yo te di autonomía en las decisiones, adiciona esas decisiones que tomaste a la bitácora».

**Propuesta previa de la IA:** Marcar como Pendiente todo lo que el equipo no hubiera revisado, y como decidido solo lo que el usuario ordenó expresamente.

**Verificación:** Se añadieron los campos **Decide** y **Alternativas descartadas**, se registraron las decisiones de la IA con su justificación (Registros 2 a 6, 8 a 16, 22 y 23) y se agregaron los registros que faltaban. Lo que dependía de una acción del equipo se dejó como Pendiente y se actualizó después (Registro 26).

**Decide:** El usuario (chat)

**Decisión:** Modificar

**Justificación:** Instrucción del usuario: la autonomía delegada hace que esas propuestas sean decisiones tomadas por la IA y no propuestas pendientes de aprobación.

### Registro 26 · Estado del trabajo del equipo y cierre de pendientes

**Fecha:** 2026-11-09

**Herramienta:** Claude Code (Anthropic); modelo de la sesión actual: Claude Sonnet 5.5

**Consulta:** El usuario pidió registrar que el equipo ya verificó las fuentes originales de las referencias y las citas, que decidió dejar las citas del Word en formato APA, que completó los enlaces y que hizo el video y colocó su enlace, y que se eliminara el resto de pendientes del equipo.

**Propuesta previa de la IA:** Mantener esos puntos como Pendiente hasta que el equipo los resolviera.

**Verificación:** Todo se registra como declaración del usuario. La IA no comprobó las fuentes, el estilo APA del archivo final, los enlaces ni el video. Se eliminó la sección «Pendientes del equipo». El antiplagio dejó de figurar en la bitácora y no se registra su resultado. El 2026-11-12 el usuario aclaró que el enlace del video aún falta en el Word (Registro 24); después lo agregó (Registro 30).

**Decide:** El usuario (chat)

**Decisión:** Modificar

**Justificación:** Instrucción del usuario: «elimina el resto de pendientes del equipo». Se actualizaron los Registros 7, 23 y 24.

### Registro 27 · Versión del equipo del informe Word

**Fecha:** 2026-11-12

**Herramienta:** Microsoft Word (equipo); revisión de solo lectura de Claude Code (modelo de la sesión actual: Claude Sonnet 5.5)

**Consulta:** El usuario informó que cargó en el repositorio el documento Word revisado y completado por el equipo (solo falta el enlace del video; pidió ignorar ese faltante) y que le hizo correcciones. Pidió a la IA revisarlo sin modificarlo.

**Propuesta:** Reemplazar la versión que generó la IA (Registro 23) por la versión del equipo, en el mismo archivo `Carvajal_Sandoval_Velasquez_Polisomnografia.docx`.

**Verificación:** Revisión de solo lectura: la IA no modificó el archivo (mismo tamaño, fecha y huella SHA-256 antes y después de revisarlo). Estado del archivo: 4.1 MB, última modificación el 2026-11-12 a las 17:08, sin versionar en Git. Contenido: portada y estilos de la plantilla institucional; las 12 figuras en el orden de los nuevos nombres de archivo (Registro 28); ecuaciones como imagen (seis) o como texto, no como ecuaciones editables de Word; lista de 31 referencias en APA, en orden alfabético, que incluye las dos de ejemplo de la plantilla (Rivas y Sánchez); Anexos A a E (A y B con resultados y código; C con un enlace de Google Drive rotulado «OneDrive»; D para el video, sin enlace; E con el enlace de GitHub). En el cuerpo del texto no hay citas APA de autor y año; quedan algunos números entre corchetes sueltos (por ejemplo [4], [7], [14], [25], [26] y [27]). El detalle de la revisión se entregó al usuario en el chat. El usuario corrigió después esas indicaciones (Registro 30).

**Decide:** El usuario (chat)

**Decisión:** Modificar

**Justificación:** Declaración del usuario: «Ya cargué el documento en Word revisado y completado» y «le hice algunas correcciones al documento». Sustituye a la versión generada por la IA, que ya no está en el repositorio.

### Registro 28 · Cambio de nombres de las figuras de `paper/figures`

**Fecha:** 2026-11-12

**Herramienta:** El usuario renombró los archivos; verificación de Claude Code (modelo de la sesión actual: Claude Sonnet 5.5)

**Consulta:** El usuario explicó que las figuras de `paper/` estaban en un orden distinto al del trabajo escrito y les cambió el nombre para ordenarlas.

**Propuesta:** Nueva numeración, de nombre anterior a nombre nuevo: `fig04_hipnograma_noche` → `fig03_hipnograma_noche`; `fig06_escenarios` → `fig04_escenarios`; `fig08_sensibilidad` → `fig05_sensibilidad`; `fig09_baum_welch` → `fig06_baum_welch`; `fig11_patologias` → `fig7_patologias`; `fig03_matrices_A_B` → `fig08_matrices_A_B`; `fig07_matrices_confusion` → `fig09_matrices_confusion`; `fig05_senales_eeg` → `fig10_senales_eeg`; `fig10_matrices_aprendidas` → `fig11_matrices_aprendidas`. `fig01_flujo_metodologico`, `fig02_grafo_hmm` y `fig12_validacion_simulador` no cambian.

**Verificación:** Los nueve archivos son renombres puros: tienen el mismo contenido que en el commit `ffe0964` (comprobado con huellas). La numeración nueva coincide con la de las figuras del Word. Git lo ve como nueve archivos borrados y nueve sin rastrear; no hay commit con el cambio. Efectos comprobados: (1) tres imágenes del README apuntan a nombres que ya no existen (`fig04_hipnograma_noche.png`, `fig06_escenarios.png` y `fig08_sensibilidad.png`); (2) `main_simulation.py` y el `main.tex` local siguen usando los nombres anteriores, así que volver a ejecutar el estudio crearía de nuevo los archivos con los nombres viejos; (3) la prueba `test_figuras_y_tablas_existen` falla (35 de 36 pasan); (4) `fig7_patologias.png` no lleva el cero inicial que usan los demás (`fig03` a `fig12`). En ese momento la IA no había modificado nada de esto; los efectos (1) a (3) se corrigieron después (Registro 29).

**Decide:** El usuario (chat)

**Decisión:** Modificar

**Justificación:** Declaración del usuario: las figuras estaban en un orden distinto al del trabajo escrito y las renombró para ordenarlas.

### Registro 29 · Actualización de las referencias a los nombres nuevos de las figuras

**Fecha:** 2026-11-12

**Herramienta:** Claude Code (modelo de la sesión actual: Claude Sonnet 5.5), con Python y Tectonic 0.17 para verificar

**Consulta:** Tras el Registro 28, la IA preguntó si actualizaba el README, `main_simulation.py` y `main.tex` a los nombres nuevos de las figuras.

**Propuesta:** Reemplazar en esos tres archivos los nueve nombres anteriores por los nuevos, sin renombrar ni mover ningún archivo de figura (`fig7_patologias.png` se dejó con el nombre que le puso el usuario) y sin tocar el documento Word.

**Verificación:** Cambios: `main_simulation.py` (9 nombres), `paper/main.tex` local (9) y `README.md` (3 imágenes y la fila de evidencia de «Cumplimiento de la actividad», que citaba `fig03`–`fig05` y `fig11`). Comprobado: los 12 nombres que citan los tres archivos existen en `paper/figures/`; las 36 pruebas pasan de nuevo; el estudio rápido (`--quick`), ejecutado en una carpeta temporal, genera exactamente los mismos 12 nombres que hay en el repositorio sin tocar `results/` ni `paper/`; `main.tex` compila con Tectonic sin avisos. Siguen sin commit `README.md`, `main_simulation.py` y el renombre de las figuras.

**Decide:** El usuario (chat)

**Decisión:** Aceptar

**Justificación:** Respuesta del usuario: «Sí, actualízalos». El cambio corrige los efectos (1) a (3) del Registro 28.

### Registro 30 · Correcciones del equipo al Word y enlace del video

**Fecha:** 2026-11-12

**Herramienta:** Microsoft Word (equipo); Claude Code solo registra el cambio (modelo de la sesión actual: Claude Sonnet 5.5)

**Consulta:** El usuario informó que corrigió en el Word las indicaciones de la revisión de la IA y que agregó el enlace del video. Pidió registrar esos cambios en la bitácora y aclaró que no es necesario que la IA revise el documento.

**Propuesta:** Las indicaciones que dio la IA en su revisión (Registro 27) fueron: citas APA de autor y año ausentes en el texto y números entre corchetes sueltos; una condición repetida en la ecuación (3); errores de texto (la llave de apertura de S, «sii» y el pie de la Figura 8); referencias (las de ejemplo de Normas APA sin citar, «Poiliou» en lugar de Pouliou y un carácter dañado en la de Khalifa); estructura (sin sección «Resumen», referencias a secciones numeradas que los títulos no tienen y «Listing» frente a «Listado»); y anexos (un rótulo «OneDrive» con enlace de Google Drive y un Listado 2 que menciona `main.tex` y el guion del pitch).

**Verificación:** La IA no la hizo: el usuario pidió expresamente no revisar el documento. Lo único comprobado es que el archivo cambió de nuevo el 2026-11-12 a las 17:33 y que sigue sin versionar en Git.

**Decide:** El usuario (chat)

**Decisión:** Aceptar

**Justificación:** Declaración del usuario: «Corregí las indicaciones que me diste en el Word, agregué el link del video».

### Registro 31 · Nombre de la figura de patologías

**Fecha:** 2026-11-12

**Herramienta:** El usuario renombró el archivo; Claude Code ajustó las referencias (modelo de la sesión actual: Claude Sonnet 5.5)

**Consulta:** En el Registro 28 la IA señaló que `fig7_patologias.png` no llevaba el cero inicial de los demás nombres. Al preparar los pasos del commit, la IA detectó que el archivo ya se llama `fig07_patologias.png`.

**Propuesta:** Mantener `fig07_patologias.png` y actualizar sus tres referencias: `main_simulation.py`, el `main.tex` local y la fila de evidencia del README.

**Verificación:** Con el renombre volvía a fallar una prueba (`test_figuras_y_tablas_existen`) porque las referencias seguían en `fig7_patologias`. Se corrigió una referencia en cada uno de los tres archivos; las 36 pruebas pasan y los 12 nombres de figura que citan existen en `paper/figures/`. Siguen sin commit.

**Decide:** El usuario (renombró el archivo) y Claude Code (autonomía delegada, para las referencias)

**Decisión:** Aceptar

**Justificación:** Todos los nombres siguen ahora el formato `figNN` en orden. Actualizar las referencias continúa el cambio que el usuario aprobó en el Registro 29 («Sí, actualízalos»).

## Acuerdos de rigor del laboratorio y evidencia

Los ocho acuerdos de rigor del laboratorio y lo que consta en el proyecto.

| Acuerdo | Estado | Evidencia |
|---|---|---|
| No inventar citas, DOI, resultados ni ejecuciones | Cumplido, con una salvedad | Las 29 referencias se comprobaron por DOI contra Crossref; las cifras del informe salen de `results/` mediante macros generadas, no se escriben a mano (Registros 6 y 14). No se verificó la indexación en Scopus/WoS. |
| Leer las fuentes originales; distinguir preprints de artículos revisados | Cumplido, según el equipo | El usuario informó el 2026-11-09 que el equipo verificó las fuentes originales y las citas (Registro 7); la IA no lo comprobó. |
| No llamar sistemática a una búsqueda exploratoria | Cumplido | El informe dice «exploratoria y dirigida, no sistemática» (Registro 8). |
| Separar datos de entrenamiento y prueba | Cumplido | Noches de prueba disjuntas de las de entrenamiento en Baum-Welch (Registro 9). |
| No reportar exactitud contra estados reales desconocidos | Cumplido | Simulación con la verdad oculta conocida; sin datos reales, lo que se declara como limitación. |
| Registrar versiones, parámetros y semillas; no repetir la misma semilla | Cumplido | `results/environment.json`, `requirements.txt` y `SeedSequence.spawn` (Registro 3). |
| No compartir datos personales sin autorización | Cumplido | No se usan datos de pacientes ni personales. |
| Conservar resultados negativos y límites del estudio | Cumplido | Se reportan Baum-Welch con inicialización aleatoria (62.1 %) y la ventaja de solo 1.3 puntos del contexto temporal con las matrices nominales (Registros 2 y 9). |

