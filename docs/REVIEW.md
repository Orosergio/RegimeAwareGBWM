# Revisión del deck — Regime-Aware GBWM with RL

_Revisión del paper, del proyecto y de la presentación web (`docs/`), con el detalle de los arreglos aplicados. Junio 2026._

---

## 1. El paper (a fondo)

**Dixon & Halperin (2020), _G-Learner and GIRL: Goal-Based Wealth Management with Reinforcement Learning_ (arXiv:2002.10990).** El paper plantea la gestión de patrimonio orientada a metas (planes de retiro, target-date funds, aportes y retiros periódicos, rebalanceo) como un MDP y lo resuelve con **G-learning**: una extensión probabilística y regularizada por entropía de Q-learning (Fox et al., 2015), pensada para datos financieros ruidosos. Su aporte clave es que, con recompensa cuadrática y política de referencia gaussiana, G-learning equivale a un **LQR regulado por entropía** — tratable y escalable a portafolios de alta dimensión. El objetivo no es el retorno sino la **probabilidad de alcanzar la meta**, P[V_T ≥ P_T] (como el precio de una opción binaria sobre la riqueza final), frente a la media-varianza de Markowitz. El paper añade además **GIRL** (la variante de RL inverso).

La presentación representa el paper con fidelidad: la slide del paper, la formulación MDP y el lenguaje ("extensión probabilística de Q-learning", "datos ruidosos", "probabilidad de meta") son correctos.

## 2. El proyecto — ¿cumple su función?

**Sí, con holgura.** El repositorio no solo reproduce el paper: lo valida y lo mejora.

- **Viable / reproducible:** reproduce el patrón de libro de texto del agente orientado a metas — arriesgar cuando vas atrás, proteger cerca de la meta — y, con regímenes, baja exposición en mal clima (bull ≈ 100% › high-vol › stable › bear ≈ 20%).
- **Comprobable:** lo mide con Monte-Carlo (4,000 paths) **y**, sobre todo, con un **backtest honesto sin look-ahead** ("desplegado en 1999") sobre historia real de S&P 500, NASDAQ, KOSPI, Nikkei, etc. `HISTORY.md` incluso autocritica el sesgo de look-ahead de un intento previo — señal de rigor poco común.
- **Mejorable:** la extensión regime-aware (cadena de Markov oculta + creencia por HMM en el estado) aporta valor justo donde importa: en metas ambiciosas mejora P(meta) y reduce shortfall **y** drawdown.
- **Respaldo de ingeniería:** ~104 funciones de test en 18 archivos, checkpoints entrenados (`g_learner`, `regime_aware_g_learner`), app Streamlit y un sandbox "Live Bank". El sandbox reproduce el backtest validado exactamente.

Cifras reales que ahora usa el deck (de `report.md` / `HISTORY.md`):

| Evidencia | Resultado |
|---|---|
| Monte-Carlo, meta default | G-Learner P(meta) 0.98 · Regime-Aware 0.94 · Glide 0.79 · 60/40 0.78 · Buy&Hold 0.63 |
| Meta ambiciosa ($550k) | Glide 0.17 → G-Learner 0.23 → **Regime-aware 0.24** (menor shortfall y drawdown) |
| Real, S&P 500 1999→2025 (meta $600k) | G-Learner $642k / 13% DD · Regime-Aware $646k / 19% DD · Buy&Hold $1,233k / **50% DD** |
| Riesgo de secuencia (empezar 2007) | Regime-aware **16%** DD vs Buy&Hold **48%** |

## 3. Diagnóstico de la web (el "caos")

1. **No se distinguía dónde estaba el paper.** El paper aparecía solo en una slide y nunca se volvía a marcar; no había sistema visual que separara "esto es del paper" de "esto es mi extensión" o "esto es evidencia".
2. **14 slides sin capítulos ni mapa.** El orden saltaba (problema → paper → curso → MDP → extensión → por qué no un bot → por qué este paper…), sin sensación de "dónde estoy".
3. **Exceso de texto** en varias slides (curso con 12 bullets, stack con 12 ítems, párrafos densos en problema/paper/alternativas/evaluación).
4. **Navegación no obvia.** Quien abría el link no sabía que se avanza con flechas/clic; sin landing, sin progreso, sin pista.
5. **Desajuste de fondo.** Estaba escrito como propuesta a futuro ("lo construiré", "resultado esperado", trayectorias _ilustrativas_), pero el proyecto ya estaba terminado con resultados reales.

## 4. Qué se arregló

**Sistema de capítulos + "tracks" de fuente (el arreglo central).** Cada slide lleva una etiqueta de color: **Paper** (teal · Dixon & Halperin), **Mi extensión** (ámbar), **Evidencia** (verde) y **Contexto** (neutro). Una **barra de progreso fija** arriba pinta las 14 slides con el color de su track, así el recorrido entero es un mapa: el paper se ve de un vistazo en teal. Se añadió una **slide de "Roadmap"** que enseña los 6 capítulos y el código de color al inicio.

**Flujo reordenado en 6 capítulos:** 1) El problema · 2) El paper (qué es + por qué este) · 3) El método (MDP + stack) · 4) Mi extensión (regímenes + por qué no un bot) · 5) La evidencia (cómo se juzga + Monte-Carlo + resultados reales) · 6) Por qué importa (usos + cierre). Se eliminó la slide redundante de "course fit" (se condensó a una línea en el Roadmap).

**Resultados reales en vez de ilustrativos.** La antigua slide de "resultado esperado" con trayectorias ilustrativas se reemplazó por (a) una tabla Monte-Carlo con números reales y (b) la **figura real del backtest 1999→2025** (`uploads/history_sp500_1999.png`) con el titular: alcanza la meta con **un tercio del drawdown** (13–19% vs 50%), recortó renta variable 65%→25% entrando a la crisis de 2008, y el dato de riesgo de secuencia 2007 (16% vs 48%). Marca explícita de honestidad: regímenes por prior económico + filtro de creencia solo con datos pasados.

**Menos texto.** Párrafos recortados a lo esencial, listas reducidas, una idea por slide.

**Navegación intuitiva.** Avance por **rueda del ratón** y **clic** (además de flechas/teclado, que se conservan), **pista de "← → · clic · scroll"** en la primera slide, y la barra/píldora de capítulo siempre orientan. En táctil se respeta el sistema de zonas del motor (sin doble avance).

**Refinado y cálido (visual).** Paleta inspirada en tus Assets (crema cálida + tinta, teal "Gulf", ámbar de atardecer, verde meta), textura de papel sutil y micro-animaciones tipo skiper-ui: entradas escalonadas con desenfoque, contador animado en la cifra clave, y _hover-lift_ en las tarjetas. Respeta `prefers-reduced-motion`.

## 5. Archivos tocados

`docs/index.html` (reestructura, tracks, resultados reales, notas del orador realineadas), `docs/styles.css` (paleta cálida + componentes nuevos), `docs/deck-hud.js` (**nuevo**: HUD de capítulos + navegación + animaciones). El motor `docs/deck-stage.js` y `docs/animations.js` **no** se modificaron. Respaldo de los originales en `docs/_backup_<fecha>/`.

## 6. Notas y pendientes

- **Verificación visual:** validé estructura y código, pero **no pude renderizar** el deck (sin navegador disponible en el entorno). Conviene abrir `docs/index.html` localmente o por GitHub Pages para una revisión visual final.
- **Idioma:** el contenido del deck sigue en inglés (audiencia CME 241). Esta revisión está en español.
- **Assets como imágenes:** elegimos "refinar y dar calidez", así que las imágenes de `/Assets` inspiraron la paleta pero no se incrustaron. Si quieres, puedo añadir una imagen hero (p. ej. el muelle/atardecer) en la portada o el cierre.
- **Idea opcional:** convertir la barra de progreso en navegación clicable por capítulos.

## 7. La web (app de Streamlit)

"La web" con los mismos problemas es la **app interactiva** `app/streamlit_app.py` (~2,000 líneas, el laboratorio que se ejecuta), distinta del deck. Ya compartía la paleta cálida y un flujo guiado **Learn → Play → Proof** (9 páginas). Le apliqué el mismo arreglo central que al deck:

- **Sistema de "tracks" de fuente + barra de progreso** (`progress_rail`): bajo la navegación, una barra mapea las 9 páginas pintadas por su track — **Paper** (azul · "Inside the AI"), **Mi extensión** (ámbar · "One journey"), **Evidencia** (verde · Compare / Time machine / Multi-asset) y **Contexto** (neutro) — con la página actual resaltada y una línea "ahora estás en: …". Así siempre sabes dónde estás y dónde está el paper.
- **Recorte de texto** en la portada (el caption del demo en vivo).
- Validado con `py_compile` (la app arranca sin errores). El motor y la lógica no se tocaron.

**Cómo ejecutarla:** `pip install -r requirements-app.txt` y luego `streamlit run app/streamlit_app.py` (se abre en `http://localhost:8501`).

**Pendiente sugerido:** una pasada completa de reducción de texto página por página (captions/`st.info`). La dejé fuera de este cambio a ciego porque conviene hacerla contigo viéndola en vivo, para no romper nada sin poder renderizarla aquí.

## 8. Auditoría de datos — ¿cada gráfica es real?

Revisado en vivo (app corriendo) + rastreo de código. **Veredicto: todas las gráficas son reales.** Hay dos tipos:

- **Cómputo real (sin datos externos):** se recalculan cada vez desde los modelos, no hay valores escritos a mano. La revisión del código no encontró ningún dato de gráfica hardcodeado/placeholder.
  - *How it works* — "Try it": consulta la política aprendida real (Stormy → 20%). ✓
  - *Inside the AI* — heatmaps de política + curvas de convergencia Q-learning/PPO. ✓
  - *Your plan / One journey* — Monte-Carlo con tus parámetros. ✓
  - *Compare plans* — barras de P(meta): Goal-based 97% · adaptativo 94% · glide 80% (coincide con el reporte). ✓
- **Datos de mercado reales (vía proveedor con caché):**
  - *Time machine* — confirmado en vivo: banner "**Loaded real S&P 500 — Jan 1999 → Dec 2025**". El scorecard da $642,460 / 13% DD · $646,371 / 19% · $1,233,198 / 50% — **idéntico al deck**. ✓✓
  - *Multi-asset* y *Live bank* — mismo proveedor (real online; sintético **claramente avisado** si estás offline).

**Integridad de datos — arreglado:** la caché tenía 27 series **sintéticas** (el stand-in offline), 6 de ellas **sin la marca `.source`** — la app las habría podido mostrar como reales sin avisar. Las moví a `data/cache/_synthetic_quarantine/` (reversible, no borradas). Ahora la caché solo contiene datos reales (^GSPC 1228, ^IXIC 2208, ^N225 13415, ^KS11 587, GC=F, SPY, TSLA…). Además declaré `yfinance` y `pandas-datareader` en `requirements-app.txt` (faltaban: un instalador limpio habría caído a sintético).

**Garantía resultante:** online, la app solo muestra datos reales; offline, muestra el stand-in sintético **siempre etiquetado "Illustrative only"** — nunca presenta datos falsos como reales.

## 9. Tipografía — quitar el "tell" de IA

El serif italic de "money goal" (Instrument Serif) gritaba *hecho-por-IA*. La skill **impeccable** lo confirma con su escaneo: las **tres** fuentes (Geist, Geist Mono, Instrument Serif) están en la lista saturada de IA, y su guía dice textual que *"un brief premium no necesita el serif expresivo que todos usan; lo más moderno es no usar la fuente que usa todo el mundo."* Cambié el sistema completo (deck + app), inspirado también en la calidez editorial de tus Assets:

- **Sans:** Geist → **Hanken Grotesk** (grotesca humanista, cálida).
- **Serif editorial (acentos italic):** Instrument Serif → **Newsreader** (serif literaria, no la de moda).
- **Mono (eyebrows/números):** Geist Mono → **IBM Plex Mono**.

Verificado en vivo en el navegador (deck `:8000` y app `:8501`): "money goal" y "goal-based" ahora se ven como italic editorial, no como plantilla de IA. Las gráficas usan la fuente neutra de matplotlib, así que no se ven afectadas. El cambio en el deck se verá en el link de GitHub al hacer push.

**Roadmap restante (de la crítica de impeccable, `.impeccable/critique/`):** reducir texto/carga cognitiva (intro repetida en captions, analogía de la montaña duplicada, 4 disclaimers, exceso de em-dashes); hacer *visible* la política (small-multiples de los 4 regímenes a la vez en vez de uno tras un selectbox); resets de estado al cambiar inputs. El P0 de "español en gráficas en inglés" ya quedó resuelto.

## 10. Demo auto-guiada, coherente con el deck y menos texto

Objetivo (tu encargo): que deck y demo cuenten **la misma historia**, que cada sección diga **qué es / por qué / qué tocar**, y que lo entienda hasta un niño.

- **Guía por sección (`demo_guide`):** bajo la navegación, cada página muestra una píldora "DECK CH.X · …" + "👉 **Try this:** …" en lenguaje simple — ata la página a su capítulo del deck y dice exactamente qué tocar, dónde y cómo. Verificado en vivo en las 9 páginas (data-driven).
- **Política visible (verificado):** "Inside the AI" ya muestra los **4 regímenes a la vez** ("One brain, four weathers…"), con rampa **RdYlBu_r apta para daltónicos**. Es la tesis hecha visible por defecto, no escondida tras un selectbox.
- **Menos texto en la portada:** quité el eyebrow redundante (lo cubre la guía), reduje "Read me first" de párrafo + 3 viñetas a una frase + 2, y el demo en vivo sube. 4 disclaimers → 2 (el global vive en el sidebar).

Todo en la app está **en vivo** (la corres con `streamlit run`). El deck cambia en el link de GitHub **al hacer push**.

## 11. Coherencia deck ↔ app

Observación válida tuya: aunque ya compartían fuentes y paleta base, el "ambiente" difería (deck = keynote oscuro/dramático; app = dashboard claro). Acerqué la app al deck (sin tocar el deck):

- **Masthead editorial** arriba de cada página de la app: banda oscura espresso con el wordmark **Regime-Aware** en degradado + *goal-based* en Newsreader italic + tag mono — el mismo ADN de la portada del deck. Da un "momento deck" en toda la app.
- **Acentos idénticos al deck** en `oklch`: verde meta `oklch(0.60 0.13 156)`, ámbar `oklch(0.73 0.15 65)`, teal `oklch(0.60 0.085 220)` — aplicados a `--accent`, chips de track y barra de progreso.
- **Fuentes y gráficas** ya compartidas (Hanken/Newsreader/IBM Plex Mono; la figura del deck sale del motor de la app).

Verificado en vivo (portada + Proof). La app sigue ligera y usable; ahora se leen como **un solo producto**. Pendiente menor opcional: refrescar el bloque del sidebar "Goal Planner" para que combine con el masthead.
