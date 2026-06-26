# Guía completa — Deck + Demo, explicado como a un niño

Esta guía tiene tres partes:

1. **Cómo demostrar que todo lo del deck funciona en la app** (guion de demo).
2. **El deck, slide por slide** (qué hay y qué significa).
3. **La app, sección por sección** (cada slider, cada palabra, cada gráfica).

La idea en una frase: **el deck cuenta la historia (la promesa); la app es la prueba en vivo de esa historia.** El deck dice "esto funciona"; en la app lo tocas y lo ves funcionar.

---

## PARTE 1 — Cómo enseñar que el deck funciona en la app

### El "camino de oro" para la demo en vivo (5–7 min)

Sigue este orden; cada parada prueba un trozo del deck. Cada página de la app trae arriba una pista **"👉 Try this"** que te dice exactamente qué tocar.

1. **Learn → How it works** — "Aquí está la idea del deck, viva." Arrastra el clima a ⛈️ Stormy y muestra cómo el número del cerebro baja. *(Prueba la slide 3 — el problema/decisión.)*
2. **Learn → Inside the AI** — "Esta es la tesis hecha visible." Enseña los **4 climas a la vez**: en Sunny hay mucho rojo (arriesga), en Stormy casi todo azul (protege). *(Prueba la slide 8 — mi extensión de regímenes.)*
3. **Proof → Compare plans** — "Aquí está la prueba #1." La barra verde más larga = el plan que más veces llega a la meta. *(Prueba las slides 10–11 — evaluación y Monte-Carlo.)*
4. **Proof → Time machine** — "Aquí está la prueba grande, con historia real." Pulsa *Run the time machine* y muestra el scorecard: llega a la meta con ~⅓ del desplome. *(Prueba la slide 12 — resultados reales.)*
5. **Proof → Proposal coverage** — "Y aquí está el mapa: cada slide → código." *(Prueba las slides 4, 5, 6, 15.)*

> Cierre de demo: "El deck les contó la historia; la app les dejó tocarla. Mismos números en ambos."

### Mapa deck → app (qué página prueba cada slide)

| Slide del deck | Dónde se prueba en la app |
|---|---|
| 1 · Portada | La app entera (la portada de *How it works*) |
| 2 · Roadmap (colores) | La navegación Learn/Play/Proof + la barra de progreso |
| 3 · El problema | **How it works** (decisión del cerebro en vivo) |
| 4 · El paper | **Proposal coverage** (referencias) + **Inside the AI** |
| 5 · Por qué este paper | **Proposal coverage** (alineación con el curso) |
| 6 · El MDP (bucle) | **Proposal coverage** (tarjetas STATE/ACTION/REWARD/MARKET) |
| 7 · Stack y algoritmos | **Proposal coverage** (tabla de métodos) |
| 8 · Mi extensión (regímenes) | **Inside the AI** (los 4 climas) + **One journey** |
| 9 · Por qué no un bot | Lo prueban **Compare/Time machine** (juzgan por meta, no por retorno) |
| 10 · Evaluación | **Compare plans** + scorecard del **Time machine** |
| 11 · Monte-Carlo | **Compare plans** (barras de probabilidad) + **Your plan** |
| 12 · Resultados reales 1999 | **Time machine** (la figura de 3 paneles + scorecard) |
| 13 · Usos reales | Los presets del sidebar (retiro/casa/colegiatura) + **Live bank** |
| 14 · Cierre | **Proposal coverage** (el mapa completo) |

### Si ellos quieren manejar la demo solos

Diles tres cosas: (1) se avanza por **fases arriba** (Learn → Play → Proof) y páginas; (2) la **barra de colores** dice dónde están y dónde está el paper (azul = paper); (3) cada página tiene **"👉 Try this"** que dice qué tocar.

---

## PARTE 2 — El deck, slide por slide (como a un niño)

1. **Portada.** La tapa. Dice el título y que esto ya está **construido y probado con mercados reales**. Mensaje: una computadora que aprende a llegar a una meta de dinero y cambia su forma de invertir según "el clima" del mercado.

2. **Roadmap (el mapa del viaje).** Enseña los 6 capítulos y, sobre todo, **el código de colores**: azul = viene del *paper*, ámbar = *mi extensión*, verde = *evidencia/prueba*, gris = contexto. Así, en cualquier slide sabes qué estás viendo.

3. **El problema.** La gente no invierte para "ganar lo máximo"; invierte para **llegar a una meta** (jubilarse, una casa, la universidad). Analogía: un bot de trading "maneja lo más rápido posible"; nuestro agente es un **GPS** que te lleva a buen puerto, aunque cambie el camino.

4. **El paper ancla.** La ciencia en la que nos paramos: Dixon & Halperin (2020), *G-Learner*. Enseñó a una computadora a manejar dinero hacia una meta con aprendizaje por refuerzo. Primero lo **copiamos fiel**, luego le agregamos algo.

5. **Por qué este paper (y no otros 3).** Comparamos 4 opciones (Deep Hedging, Optimal Stopping, FinRL y esta). Ganó goal-based porque es **realista, factible y deja espacio para una mejora propia**.

6. **El MDP (el bucle).** Cómo "piensa" la compu, en círculo, cada mes: mira la **situación** (estado) → elige **cuánto arriesgar** (acción) → el **mercado** se mueve y entra tu aporte → recibe un **puntaje** (recompensa) → repite.

7. **Stack y algoritmos.** La caja de herramientas: la teoría (MDP, Q/G-learning, PPO) y el código (Python, NumPy, PyTorch, Gymnasium…). Construido como un **experimento controlado**, no un cuaderno desordenado.

8. **Mi extensión: los regímenes (el clima).** La idea nueva: el agente no debe manejar igual con cualquier clima. Lee el **clima del mercado** y cambia el riesgo: casi 100% en acciones cuando está **soleado** (bull), ~20% cuando está **tormentoso** (bear).

9. **Por qué no un bot de trading.** Los bots llamativos son fáciles de "trucar" y difíciles de defender. El nuestro tiene un objetivo **claro y honesto**: llegar a la meta sin sustos grandes.

10. **Evaluación (cómo se califica).** No medimos "quién ganó más dinero", sino **¿llegó a la meta?**, qué tan feo falla si falla, el **peor desplome**, cuánto rota, y si **cambia con el clima**.

11. **Monte-Carlo (prueba #1).** Jugamos **4,000 futuros simulados**. Nuestros planes llegan a la meta mucho más seguido: G-Learner **0.98**, regime-aware **0.94**, vs glide 0.79, 60/40 0.78, comprar-y-aguantar 0.63.

12. **Resultados reales (prueba grande).** Historia real desde **1999, sin trampa** (sin mirar el futuro). Llega a la meta con **un tercio del desplome**: 13–19% vs **50%** de comprar-y-aguantar. Recortó acciones 65%→25% entrando a la crisis de 2008.

13. **Usos en la vida real.** Robo-advisor que conoce tu meta, planificador de retiro, fondos target-date, app de finanzas personales, simulador educativo.

14. **Cierre.** Una frase: construir un agente que **llega a la meta y se adapta al clima**. Más una rejilla con las 6 razones por las que el proyecto es sólido.

---

## PARTE 3 — La app, sección por sección (como a un niño)

### El "clima del mercado" (esto aparece en toda la app)

- ☀️ **Sunny = Bull** → el mercado sube. El cerebro **arriesga más** (más acciones).
- 🌤️ **Calm = Stable** → mercado normal. El cerebro va **balanceado**.
- 🌧️ **Choppy = High-vol** → mercado nervioso/saltarín. El cerebro **mueve menos** y va con cuidado.
- ⛈️ **Stormy = Bear** → el mercado baja. El cerebro **protege** (casi todo en efectivo).

### El sidebar "Goal Planner" (la barra izquierda — tus números)

Estos números alimentan *Your plan, One journey* y *Compare plans*:

- **What are you saving for?** — un *preset* (retiro, casa, colegiatura, fondo de emergencia) que rellena los números por ti.
- **Money you have now ($)** — con cuánto empiezas.
- **Goal amount ($)** — la meta a la que quieres llegar.
- **Years until you need it** — cuántos años tienes (deslizador).
- **Monthly savings ($)** — cuánto agregas cada mes.
- **Advanced assumptions** (desplegable) — *Safe-cash yearly return*: cuánto rinde el efectivo seguro; *How 'sticky' market moods are*: qué tanto dura un clima antes de cambiar.

> *(Nota: el sidebar se ve un poco anticuado; te ofrezco refrescarlo después.)*

### Arriba de cada página

- **Barra de colores (progreso):** mapa de las 9 páginas pintadas por su "track" (azul = del paper, ámbar = mi extensión, verde = evidencia, gris = contexto). La actual va resaltada + "X / 9".
- **Píldora "DECK CH.X"** + **"👉 Try this: …"** — te dice a qué capítulo del deck corresponde y qué tocar.

### Las 9 páginas

**LEARN**

1. **How it works (Empieza aquí).** La idea en una pantalla, con un **cerebro vivo**. Controles: *Market weather* (elige el clima), *Where you stand* (% de la meta que ya tienes) y *Years left*. Resultado: **"THE BRAIN'S DECISION"** = cuánto pondría en acciones, con un *"Why X%?"* que lo explica. Cambia **solo el clima** y verás moverse el número: eso es lo regime-aware. *(= slide 3.)*

2. **Inside the AI (Dentro del cerebro).** Controles: *Show the learned policy of* (Smart adaptive / Goal-based / Q-learning). Gráficas:
   - **Mapa de política (4 climas):** "One brain, four weathers". Cada cuadro es una situación: tu saldo (vertical) en un momento (horizontal). **Rojo = más acciones (arriesga), azul = más efectivo (protege).** Compara el mismo cuadro entre paneles: en Stormy el rojo se encoge. *Esa diferencia ES la tesis.* (Rampa apta para daltónicos.)
   - **Curva de Q-learning:** la línea negra (el agente que aprende a prueba y error) sube hacia la verde (la solución exacta). Muestra que **aprende solo**. *(= slides 6 y 8.)*

**PLAY**

3. **Your plan (Tu plan).** Usa tus números del sidebar. Arriba 4 tarjetas: *You have now / Goal / Time / Saving*. Corre todos los planes en los mismos futuros y muestra **"Our recommendation"**: el mejor plan + tu **% de éxito** + cuántos puntos gana vs el glide path y vs comprar-y-aguantar, y el peor desplome. *(= slide 11.)*

4. **Live bank (NT$) (Banco de juego).** Un banco que **manejas tú**, con dinero de mentira en NT$. Abres una o varias *wallets*, **depositas/retiras**, y avanzas mes a mes mientras el cerebro invierte. Gráficas: **saldo en el tiempo**, **barra de reparto** (acciones vs efectivo), **dinero por activo**, una **tabla de líderes** de wallets, y un **registro de decisiones** que dice qué vio, cuánto movió y por qué. *(Muestra el modelo "en acción".)*

5. **One journey (Un viaje).** Sigues **UN solo futuro** mes a mes. Controles: *Plan to follow* y *Try a different future* (cambia la semilla del azar) y *Peek at a month*. Gráficas: la **trayectoria** del saldo, el **reparto en el tiempo** (cómo sube/baja el % en acciones), y la **creencia de régimen** (qué clima cree que hay). *(= slide 8, paso a paso.)*

**PROOF**

6. **Compare plans (Comparar planes).** La carrera justa: cada plan en los **mismos miles de futuros**. Gráfica: barras de **probabilidad de llegar a la meta** (la verde más larga gana). *(= slides 10–11.)*

7. **Time machine (Máquina del tiempo).** La prueba honesta sobre **historia real**, sin mirar el futuro. Controles: *Market* (S&P 500, NASDAQ, KOSPI…), *Start year*, *Goal for this run*, *Offline*. Botón **Run the time machine**. Resultado:
   - **Figura de 3 paneles:** (arriba) el saldo de cada plan con bandas de crisis; (medio) cómo el agente **sube/baja el % en acciones** mientras comprar-y-aguantar no se mueve; (abajo) su **creencia de "mal clima"** en tiempo real.
   - **Scorecard:** tabla con saldo final, **¿llegó?**, **peor desplome**, peor año, crecimiento. (Aquí salen $642k/13%, $646k/19%, $1,233k/50%.)
   - **"What the agent did at the turning points":** qué hizo en dot-com, 2008, COVID, 2022.
   - **Sequence-of-returns risk:** desplome según el año en que empezaste. *(= slide 12.)*

8. **Multi-asset (Cuatro activos).** Misma prueba honesta pero repartiendo entre **S&P + internacional + bonos + oro**. Botón **Split my money live**. Gráficas: **cómo se reparte** en el tiempo, **tu dinero** sobre la historia real, **scorecard**, y un **deslizador** para ver el reparto mes a mes. *(= slide 12, versión multi-activo.)*

9. **Proposal coverage (Cobertura del deck).** El mapa **deck → código**: las tarjetas del **MDP (slide 6)**, la tabla de **alineación con el curso (slide 5)**, la **cobertura slide-por-slide** y las **referencias (slide 15)**. Aquí demuestras, literal, que cada parte del deck existe como código que corre. *(= slides 4, 5, 6, 15.)*

---

## Glosario relámpago

- **Meta / goal:** el monto al que quieres llegar (p. ej. $250k).
- **P(goal) / probabilidad de éxito:** de muchos futuros, en cuántos llegaste a la meta.
- **Drawdown / desplome:** la peor caída de pico a valle por el camino (más chico = más tranquilo).
- **Shortfall:** si fallas, qué tan por debajo de la meta quedaste.
- **Regime / régimen (clima):** el estado del mercado (sunny/calm/choppy/stormy).
- **No look-ahead / sin mirar el futuro:** el agente solo ve el pasado en cada mes; nada de hacer trampa.
- **G-Learner / Q-learning / PPO:** las "recetas" de aprendizaje (la del paper, la clásica, y la de redes neuronales).
