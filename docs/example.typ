#import "@preview/subpar:0.2.2"
#import "@preview/cetz:0.4.2"
#import "@preview/finite:0.5.1": automaton
#import "@preview/lovelace:0.3.0": pseudocode-list

#let color-palette = (
  primary: rgb("#164863"),
  secondary: rgb("#2A6A8C"),
  accent: rgb("#F7693B"),
  // text: rgb("#1f2937"),
  text: rgb("#111111"),
  background: rgb("FFFFFF"),
  highlight: rgb("#FFE964"),
)

#set document(title: "Apuntes de Predicción Estructurada Estadística")

#set text(font: "Source Sans 3", size: 10pt, weight: "regular", fill: color-palette.text, number-type: "lining", lang: "es")

#set par(justify: true, leading: 6pt, spacing: 12pt)

#show link: it => underline(
  stroke: color-palette.accent,
  offset: 1.2pt,
  text(fill: color-palette.accent, weight: "medium", it),
)
#show ref: set text(weight: 600, fill: color-palette.secondary)
#show footnote: set text(weight: 600, fill: color-palette.secondary)
//
// Nivel 1 = "Tema N." (capítulos); el resto, "1.1.", "1.1.1.", ...
#set heading(numbering: (..n) => {
  let n = n.pos()
  if n.len() == 1 { "Tema " + str(n.first()) + "." } else { numbering("1.", ..n) }
})
#show heading: set block(below: 12pt)
#show heading: smallcaps
#show heading.where(level: 1): set text(18pt, weight: "semibold", fill: color-palette.text, number-type: "old-style")
#show heading.where(level: 2): set text(14pt, weight: "semibold", fill: color-palette.text, number-type: "old-style")
#show heading.where(level: 3): set text(12pt, weight: "semibold", fill: color-palette.text, number-type: "old-style")

// Los encabezados de nivel 1 numerados son "temas" (capítulos): empiezan en página nueva
#show heading.where(level: 1): it => {
  if it.numbering == none or not it.outlined { return it }
  pagebreak(weak: true)
  v(1.5cm)
  block(below: 0.6em, text(13pt, weight: "semibold", fill: color-palette.secondary, number-type: "lining")[
    #smallcaps[Tema #counter(heading).display("1")]
  ])
  block(below: 0.8em, text(22pt, weight: "semibold", fill: color-palette.text, hyphenate: false)[#smallcaps(it.body)])
  line(length: 100%, stroke: 0.9pt + color-palette.primary)
  v(0.8cm)
}
//
#set highlight(fill: color-palette.highlight, radius: 5pt, extent: 0.2em)
//
#set underline(stroke: 1pt + color-palette.secondary, offset: 2pt, evade: true, extent: 1pt)
//
#let accent-strong(body) = strong(text(fill: color-palette.accent, body))
#show strong: set text(color-palette.text)

//
#set list(marker: ([•], [--], []), indent: 6pt)

#show figure.caption: it => align(left)[
  #text(size: 10pt, weight: 600)[
    #it.supplement
    #context it.counter.display(it.numbering) :
  ]
  #text(size: 10pt)[#it.body]
]

#let note(
  title: none,
  palette: (
    title-text: color-palette.secondary,
    stroke: color-palette.primary,
    background: color-palette.background.lighten(-7%)
  ),
  body,
) = {
  v(5pt)
  block(
    width: 100%,
    height: auto,
    breakable: false,
    inset: 0.3em,
    outset: 0.3em,
    fill: palette.background,
    stroke: (
      left: (
        thickness: 5pt,
        paint: palette.stroke,
        dash: "solid"
      )
    )
  )[

    #set par(leading: 5.6pt, spacing: 10pt)
    #pad(
      left: 0.3em,
      right: 0.3em,
      strong(
        text(
          size: 9pt,
          smallcaps(title)
        )
      )
      + block(
          above: 0.8em,
          text(
            size: 9pt,
            body),
          ),
    )
  ]
  v(5pt)
}




// --------------------------------------------------------------------------------------------------
// --------------------------------------- HELPERS PARA APUNTES -------------------------------------
// --------------------------------------------------------------------------------------------------

// Derivación con símbolo encima de la flecha: $alpha dto(*) beta$, $S dto(+) x$, $dto(r_1)$
#let dto(s) = math.attach(math.limits(math.arrow.r.double), t: s)

// Independencia (condicional) y su negación: $a indep b | c$, $a nindep b$
#let indep = math.class("relation", "⫫")
#let nindep = math.class("relation", "⫫̸")

// Estilo común para los autómatas dibujados con `finite`
#let automaton-style = (
  transition: (label: (angle: 0deg, size: 0.9em)),
  state: (radius: 0.55, label: (size: 0.95em)),
)

// Diagrama "entrada -> caja -> salida" (con modelo opcional debajo)
#let pred-diagram(
  x: $bold(x)$,
  y: $bold(y)$,
  box-label: [Predicción],
  in-label: [observación],
  out-label: [hipótesis],
  model: none,
  out-len: 2.6,
) = align(center, block(above: 1em, below: 1.2em, cetz.canvas({
  import cetz.draw: *
  let arrow = (mark: (end: "stealth", fill: black, scale: 0.8), stroke: 0.6pt)
  line((0, 0), (2.6, 0), ..arrow)
  rect((2.6, -0.45), (4.8, 0.45), stroke: 0.6pt, name: "box")
  content("box", box-label)
  line((4.8, 0), (4.8 + out-len, 0), ..arrow)
  content((1.3, 0.12), x, anchor: "south")
  content((4.8 + out-len / 2, 0.12), y, anchor: "south")
  if in-label != none { content((1.3, -0.12), in-label, anchor: "north") }
  if out-label != none { content((4.8 + out-len / 2, -0.12), out-label, anchor: "north") }
  if model != none {
    line((3.7, -1.1), (3.7, -0.45), ..arrow)
    content((3.7, -1.15), model, anchor: "north")
  }
})))

// Árbol sintáctico a partir de listas anidadas: ([S], ([NP], [...]), ([VP], ...))
#let syntax-tree(t, spread: 1.0, grow: 0.85) = cetz.canvas({
  import cetz.draw: *
  cetz.tree.tree(
    t,
    spread: spread,
    grow: grow,
    draw-node: (node, ..) => content((), box(inset: 2.5pt, node.content), name: "content"),
    draw-edge: (from, to, ..) => line(from + ".content.south", to + ".content.north", stroke: 0.6pt),
  )
})

// Símbolo terminal resaltado en árboles
#let term(body) = text(fill: color-palette.accent, body)

// Grafo (dirigido o no) para modelos gráficos.
//   nodes:   (nombre: posición)
//   edges:   array de pares (origen, destino)
//   labels:  (nombre: contenido) para etiquetas distintas del nombre
//   shaded:  nodos observados (se rellenan)
//   plain:   nodos sin círculo (p. ej. "...")
#let graph-diagram(
  nodes,
  edges,
  directed: false,
  labels: (:),
  shaded: (),
  plain: (),
  radius: 0.32,
  line-stroke: 1pt,
  label-size: 10pt,
) = cetz.canvas({
  import cetz.draw: *
  let node-r(n) = if n in plain { 0.22 } else { radius }
  // Recorta el segmento p->q para que empiece en el borde del nodo p
  let shorten(p, q, r) = {
    let (x1, y1) = p
    let (x2, y2) = q
    let d = calc.sqrt(calc.pow(x2 - x1, 2) + calc.pow(y2 - y1, 2))
    (x1 + r * (x2 - x1) / d, y1 + r * (y2 - y1) / d)
  }
  for (a, b) in edges {
    let (pa, pb) = (nodes.at(a), nodes.at(b))
    line(
      shorten(pa, pb, node-r(a)),
      shorten(pb, pa, node-r(b)),
      stroke: line-stroke,
      mark: if directed { (end: "stealth", fill: black, scale: 0.7) } else { none },
    )
  }
  for (name, pos) in nodes {
    if name not in plain {
      circle(
        pos,
        radius: radius,
        fill: if name in shaded { color-palette.secondary.lighten(70%) } else { white },
        stroke: line-stroke,
      )
    }
    content(pos, text(size: label-size, labels.at(name, default: name)))
  }
})

// Grafo no dirigido de ejemplo (estilo de las transparencias)
#let undirected-graph(nodes, edges, radius: 0.32) = graph-diagram(
  nodes, edges, radius: radius, line-stroke: 1.2pt, label-size: 12pt,
)

// --------------------------------------------------------------------------------------------------
// ------------------------------------------- FRONT PAGE -------------------------------------------
// --------------------------------------------------------------------------------------------------

#let eyebrow = "Predicción Estructurada Estadística"
#let studies = "Máster en Inteligencia Artificial, \nReconocimiento de Formas e Imagen Digital"
#let title = "Apuntes de la asignatura"
#let subtitle = "Curso 2026-2027"


#set page(
  paper: "a4",
  margin: (left: 2cm, right: 2cm, top: 2.5cm, bottom: 2cm),
  numbering: none,
  fill: color-palette.background,
)

#set align(center)

#block(
  width: 100%,
  height: 100%,
  [
    #figure(
      image("assets/logo_horizontal_UPV.png", height:6cm),
      numbering: none,
    )
    #v(1.5cm)
    //---------------------------------------------------------
    #text(size: 19pt, weight: 500, fill: color-palette.text)[
      #smallcaps(eyebrow)
    ]
    #v(5pt)
    #text(size: 15pt, weight: 400, fill: color-palette.text)[
      #smallcaps(studies)
    ]
    #v(30pt)
    #line(length: 100%, stroke: 1.5pt + color-palette.text)
    #v(5pt)
    #text(size: 26pt, weight: "bold", fill: color-palette.text)[
      #smallcaps(title)
    ]
    #v(5pt)
    #text(size: 20pt, weight: "medium", fill: color-palette.text)[
      #smallcaps(subtitle)
    ]
    #v(5pt)
    #line(length: 100%, stroke: 1.5pt + color-palette.text)
    #v(2cm)
    #text(size: 16pt, weight: "medium", fill: color-palette.text)[
      David González Durán
    ]
    #v(1pt)
    #text(size: 12pt, weight: "medium", fill: color-palette.text)[
      (dgondur\@posgrado.upv.es)
    ]

    #v(2.3cm)

    #figure(
      image("assets/logo_DSIC.png", height: 2cm),
      numbering: none,
    )
    #text(size: 12pt, weight: "medium", fill: color-palette.text)[
      #smallcaps("Departamento de Sistemas \nInformáticos y Computación")
    ]

    #line(length: 40%, stroke: 1pt + color-palette.text)

    #text(size: 12pt, weight: "semibold", fill: color-palette.text)[#smallcaps("Valencia, curso 2026-2027")]
  ]
)

#pagebreak()
#set align(left)

// --------------------------------------------------------------------------------------------------
// -------------------------------------- AUTHORITY DECLARATION -------------------------------------
// --------------------------------------------------------------------------------------------------

// #set page(
//   margin: (left: 1.8cm, right: 1.8cm, top: 2.2cm, bottom: 2cm),
//   numbering: "I",
//   number-align: bottom + right,
// )
//
// #counter(page).update(1)
//
// #set align(left)
//
// #heading(level: 1, numbering: none, outlined: false)[Declaración de autoridad y uso de inteligencia artificial generativa]
//
// #pagebreak()

// --------------------------------------------------------------------------------------------------
// ---------------------------------------- TABLE OF CONTENTS ---------------------------------------
// --------------------------------------------------------------------------------------------------

#show outline.entry.where(level: 1): set block(above: 1.5em, below: 0.5em)
#show outline.entry.where(level: 2): set block(above: 1.0em, below: 0.5em)
#show outline.entry.where(level: 3): set block(above: 1.0em, below: 0.5em)

#outline(
  title: [Índice general #v(10pt)],
  depth: 2,
  indent: 0.5cm
)

#pagebreak()
#counter(page).update(1)

// --------------------------------------------------------------------------------------------------
// ---------------------------------------------- BODY ----------------------------------------------
// --------------------------------------------------------------------------------------------------

#set page(
  margin: (left: 1.8cm, right: 1.8cm, top: 2.4cm, bottom: 2cm),

  numbering: "1",
  number-align: bottom + center,

  // Cabecera para el resto de páginas (páginas > 1)
  header: context {
    let page-num = counter(page).get().first()

    // ¿Empieza un tema en esta página? Entonces no se pone cabecera
    let chapter-here = query(heading.where(level: 1)).any(h => h.numbering != none and h.location().page() == here().page())

    if page-num > 1 and not chapter-here {
      // Busca el heading último (de cualquier nivel) actual
      let headings = query(selector(heading).before(here()))

      // Extrae el último heading encontrado (si existe)
      let current-heading = if headings.len() > 0 {
        let h = headings.last()
        // Verificamos si el heading tiene numeración asignada
        if h.numbering != none {
          // Obtenemos la numeración completa adaptada al nivel (ej. "1.2" o "1.2.3")
          let num = numbering(h.numbering, ..counter(heading).at(h.location()))
          [#num #h.body]
        } else {
          h.body
        }
      } else {
        []
      }

      // Grid superior con el título a la izquierda y el número de página a la derecha
      grid(
        columns: (1fr, auto),
        text(size: 10pt, weight: 500, fill: color-palette.text)[#current-heading],
        text(size: 10pt, weight: 500, fill: color-palette.text)[#counter(page).display()]
      )
      v(-4pt)
      line(length: 100%, stroke: 0.9pt + color-palette.text)
    }
  },

  // Pie de página solo para la página 1 (centrado abajo)
  footer: context {
    let page-num = counter(page).get().first()

    if page-num == 1 {
      align(center, text(size: 10pt, weight: 500, fill: color-palette.text)[#counter(page).display()])
    }
  }
)


//---------------------------------------------------------------------------------------

= Introducción <sec:intro>

La asignatura se organiza en cuatro bloques:

+ Introducción.
+ Modelos para la predicción estructurada estadística.
+ Predicción: decodificación e inferencia.
+ Estimación de los parámetros de los modelos.

En este primer tema se presenta el problema de la *predicción estructurada*, se motiva con aplicaciones reales y se introducen los dos formalismos que servirán de base a lo largo del curso: los *_(Weighted) Finite-State Automata_* para predecir secuencias y las *_Context-Free Grammars_* para predecir árboles.

== Predicción no estructurada <sec:pred_no_estructurada>

#pred-diagram()

En el problema clásico de predicción (o *predicción no estructurada*) intervienen tres elementos:

- *Observación de entrada*: $bold(x) in cal(X)$, que puede ser cualquier tipo de objeto.
- *Hipótesis de salida*: $y in cal(Y)$ es un número (real o entero): $cal(Y) = {1, dots, K}$ o $y in RR$.
- *Función de predicción (no estructurada)*: $f : cal(X) -> RR$, que asigna una hipótesis $y = f(bold(x))$ a cada entrada $bold(x)$. Según el tipo de salida tenemos:
  - Clasificación binaria: $y in {0, 1}$.
  - Clasificación multiclase: $y in {1, dots, K}$.
  - Regresión: $y in RR$.

#note(title: "Pregunta clave")[
  ¿Qué ocurre si el espacio de salidas es mucho más grande y además *tiene estructura*? Por ejemplo, si las salidas son secuencias, árboles o, en general, grafos.
]

== Predicción estructurada <sec:pred_estructurada>

=== Definición

#pred-diagram()

- *Observación de entrada*: $bold(x) in cal(X)$, que puede ser cualquier tipo de objeto.
- *Hipótesis de salida*: $bold(y) in cal(Y)$ es un *objeto estructurado complejo*.
- *Función de predicción (estructurada)*: $f : cal(X) -> cal(Y)$, que asigna una hipótesis $bold(y) = f(bold(x))$ a cada entrada $bold(x)$, donde $bold(y)$ puede ser una secuencia, un árbol de análisis, un grafo...

=== Aplicaciones

La predicción estructurada aparece en multitud de problemas reales, en los que la salida es un objeto complejo (en paréntesis la el tipo de salida esperado):

- *Procesamiento del lenguaje natural*: etiquetado morfosintáctico (_PoS tagging_) (frases), análisis sintáctico (árboles), traducción automática (frases o grafos de palabras), extracción de información (frases).
- *Procesamiento de imágenes*: análisis de escenas (frases o grafos de relaciones), reconocimiento de texto manuscrito (frases o grafos de palabras).
- *Procesamiento del habla*: transcripción automática (frases o grafos de palabras), síntesis de voz (señal de audio).
- *Bioinformática*: predicción de la estructura de proteínas (árboles o grafos).
- *Robótica*: planificación (secuencias de acciones).



=== Espacio de búsqueda

#pred-diagram(x: $bold(x) in cal(X)$, y: $bold(y) in cal(Y)$, model: $M$)

Típicamente $cal(Y)$ es *potencialmente infinito*, por lo que calcular $f(bold(x)) = bold(y) in cal(Y)$ será a menudo *intratable*.

#note(title: "Solución estructural")[
  Se asume una estructura concreta en $cal(Y)$ y se define un *modelo* $M$ que permite caracterizar dicha estructura, restringiendo el espacio de búsqueda a $Y(M) subset.eq cal(Y)$:
  $ f(bold(x)) = bold(y) in Y(M) $
  Dependiendo de $M$, es posible encontrar soluciones en *tiempo polinómico*.
]

== Predicción de secuencias <sec:pred_secuencias>

Muchos objetos de interés son secuencias:
- un texto es una secuencia de palabras (o incluso de letras),
- una locución hablada es una secuencia de vectores acústicos extraídos de una señal de voz,
- un vídeo es una secuencia de fotogramas, ...

Los modelos de estados finitos son la herramienta natural para representarlas:

- *_Finite-State Acceptors_*: representaciones compactas de conjuntos regulares, eficientes de explorar. P. ej. búsqueda de patrones, tokenización o compresión.
- *_Finite-State Transducers_*: representaciones compactas de relaciones binarias racionales, eficientes de explorar y de combinar. P. ej. diccionarios o reglas dependientes del contexto.
- *_Weighted Automata_*: los pesos codifican típicamente incertidumbre (probabilidades). P. ej. modelos de lenguaje de $n$-gramas o _Hidden Markov Models_ (HMM).

=== _Weighted Finite-State Automaton_ (WFSA) <sec:wfsa>

#note(title: [Definición: _Weighted Finite-State Automaton_ (WFSA)])[
  Un *_Weighted Finite-State Automaton_* (WFSA) es una tupla $A = (Sigma, Q, I, F, delta, lambda, rho)$ donde:
  - $Sigma$ es un alfabeto finito de entrada.
  - $Q$ es el conjunto de estados, $I subset.eq Q$ los estados iniciales y $F subset.eq Q$ los finales.
  - $delta subset.eq Q times (Sigma union {epsilon}) times KK times Q$ es la función (relación) de transición.
  - $lambda : I -> KK$ es la función de peso inicial y $rho : F -> KK$ la de peso final.

  $KK$ denota el conjunto de pesos (los de un semianillo), y ${epsilon}$ es la cadena vacía.
]

En la @fig:wfsa, cada transición se etiqueta como _símbolo/peso_ y cada estado final como _estado/peso final_.

#figure(
  automaton(
    (
      q0: (q1: "x", q2: "y"),
      q1: (q3: "z"),
      q2: (q3: "w"),
      q3: none,
    ),
    initial: "q0",
    final: ("q1", "q2", "q3"),
    labels: (
      q0: [0], q1: [1/2], q2: [2], q3: [3/2],
      q0-q1: [a/1 \ b/4],
      q0-q2: [a/2 \ b/1],
      q1-q3: [b/1 \ c/3],
      q2-q3: [b/3 \ c/5],
    ),
    layout: (q0: (0, 0), q1: (3.2, 1.3), q2: (3.2, -1.3), q3: (6.4, 0)),
    style: automaton-style + (
      q0: (initial: (label: none)),
      q0-q1: (curve: 0.4, label: (dist: 0.5)),
      q1-q3: (curve: 0.4, label: (dist: 0.5)),
      q0-q2: (curve: -0.4, label: (dist: -0.5)),
      q2-q3: (curve: -0.4, label: (dist: -0.5)),
    ),
  ),
  caption: [_Weighted Finite-State Automaton_ (WFSA). El estado 0 es inicial; los estados 1, 2 y 3 son finales con pesos finales 2, $bold(1)$ y 2 respectivamente (cuando no se indica el peso final, es el elemento neutro del producto del semianillo).],
) <fig:wfsa>


== Análisis sintáctico <sec:parsing>

El análisis sintáctico (_syntactic parsing_) consiste en obtener la estructura sintáctica (un árbol) asociada a una frase. Por ejemplo, para la frase del _Penn Treebank_ @marcus1993building

#align(center)[_“Pierre Vinken, 61 years old, will join the board as a nonexecutive director Nov. 29.”_]

la anotación en formato de corchetes es

#block(inset: (x: 1em))[
  #set text(size: 8.5pt)
  `((S (NP-SBJ (NP (NNP Pierre) (NNP Vinken)) (, ,) (ADJP (NP (CD 61) (NNS years)) (JJ old)) (, ,)) (VP (MD will) (VP (VB join) (NP (DT the) (NN board)) (PP-CLR (IN as) (NP (DT a) (JJ nonexecutive) (NN director))) (NP-TMP (NNP Nov.) (CD 29)))) (. .)))`
]

que corresponde al árbol de la @fig:penn_tree.

#figure(
  {
    set text(size: 7pt)
    syntax-tree(spread: 0.5, grow: 0.75,
      ([S],
        ([NP-SBJ],
          ([NP], ([NNP], [Pierre]), ([NNP], [Vinken])),
          ([,], [,]),
          ([ADJP], ([NP], ([CD], [61]), ([NNS], [years])), ([JJ], [old])),
          ([,], [,]),
        ),
        ([VP],
          ([MD], [will]),
          ([VP],
            ([VB], [join]),
            ([NP], ([DT], [the]), ([NN], [board])),
            ([PP-CLR], ([IN], [as]), ([NP], ([DT], [a]), ([JJ], [nonexecutive]), ([NN], [director]))),
            ([NP-TMP], ([NNP], [Nov.]), ([CD], [29])),
          ),
        ),
        ([.], [.]),
      )
    )
  },
  caption: [Árbol de análisis sintáctico de una frase del _Penn Treebank_.],
) <fig:penn_tree>

=== El análisis sintáctico como problema de búsqueda

#pred-diagram(
  x: $x in Sigma^*$,
  y: $t in cal(T) : "yield"(t) = x in L(G)$,
  box-label: [_Parsing_],
  in-label: none, out-label: none,
  model: $G$,
  out-len: 4.6,
)

Dada una cadena $x$ y una gramática $G$, se busca un árbol $t$ del conjunto de árboles $cal(T)$ cuya *frontera* ($"yield"(t)$, la secuencia de hojas leída de izquierda a derecha) sea $x$.

Los *modelos gramaticales* se usan porque:
- son el modelo más simple y natural para estructuras arbóreas;
- su marco formal (matemático) es bien conocido;
- son modelos compactos (pocos parámetros libres);
- se comportan bien frente al problema de la ambigüedad;
- los modelos incontextuales representan bien las dependencias a largo plazo de las restricciones sintácticas y semánticas del lenguaje natural.

=== _Context-Free Grammars_ (CFG) <sec:cfg>

#note(title: [Definición: _Context-Free Grammar_ (CFG)])[
  Una *_Context-Free Grammar_* (CFG) es una tupla $G = (Sigma, N, S, cal(P))$ donde:
  - $Sigma$ es el conjunto de *símbolos terminales*.
  - $N$ es el conjunto de *símbolos no terminales* (o variables), con $N inter Sigma = emptyset$.
  - $S in N$ es el *símbolo inicial* (axioma).
  - $cal(P)$ es el conjunto de *reglas* (o producciones): $(A -> alpha) in cal(P)$, con $A in N$ y $alpha in (N union Sigma)^*$.
]

- *Derivación directa*: $delta A gamma$ *deriva directamente* $delta alpha gamma$:
  $ delta A gamma => delta alpha gamma quad "sii" quad exists (A -> alpha) in cal(P); quad delta, gamma in (N union Sigma)^* $
- *Derivación*: $alpha$ *deriva* $beta$:
  $ alpha dto(*) beta quad "sii" quad exists alpha_0, dots, alpha_m in (N union Sigma)^* : alpha = alpha_0 => alpha_1 => dots => alpha_(m-1) => alpha_m = beta $
- *Lenguaje generado por una gramática*:
  $ L(G) = {x | x in Sigma^* : S dto(+) x} $
- *Teorema*:
  $ x in L(G) quad "sii" quad S dto(+) x quad "sii" quad exists t in cal(T) : "yield"(t) = x $

==== Ejemplo

Gramática sencilla tomada de @manning1999foundations:

#figure(
  table(
    columns: 4,
    align: left,
    stroke: 0.5pt,
    inset: (x: 8pt, y: 4pt),
    [$S -> "NP" "VP"$], [$"VP" -> V "NP"$], [$V -> mono("saw")$], [$"NP" -> mono("saw")$],
    [$"NP" -> "NP" "PP"$], [$"VP" -> "VP" "PP"$], [$"NP" -> mono("astronomers")$], [$"NP" -> mono("stars")$],
    [$"PP" -> P "NP"$], [$P -> mono("with")$], [$"NP" -> mono("ears")$], [$"NP" -> mono("telescopes")$],
  ),
  caption: [Reglas de una _Context-Free Grammar_ (CFG) sencilla.],
) <tab:cfg_example>

Una posible derivación (por la izquierda) de la frase _astronomers saw stars with telescopes_ es:

$
  S & => "NP" "VP" => mono("astronomers") "VP" => mono("astronomers") V "NP" => mono("astronomers saw") "NP" \
  & => mono("astronomers saw") "NP" "PP" => mono("astronomers saw stars") "PP" => mono("astronomers saw stars") P "NP" \
  & => mono("astronomers saw stars with") "NP" => mono("astronomers saw stars with telescopes")
$

La frase es *ambigua*: admite los dos árboles de la @fig:cfg_trees (el sintagma preposicional _with telescopes_ puede modificar a _stars_ o a _saw_).

#figure(
  grid(
    columns: 2,
    column-gutter: 1.5cm,
    align: bottom,
    syntax-tree(
      ([S], ([NP], [astronomers]), ([VP], ([V], [saw]), ([NP], ([NP], [stars]), ([PP], ([P], [with]), ([NP], [telescopes])))))
    ),
    syntax-tree(
      ([S], ([NP], [astronomers]), ([VP], ([VP], ([V], [saw]), ([NP], [stars])), ([PP], ([P], [with]), ([NP], [telescopes]))))
    ),
  ),
  caption: [Los dos árboles de análisis de _astronomers saw stars with telescopes_.],
) <fig:cfg_trees>

=== Algoritmo CKY <sec:cky>

El algoritmo de *Cocke-Kasami-Younger* (CKY) resuelve el problema de pertenencia ($x in L(G)$?) mediante programación dinámica, para gramáticas en *forma normal de Chomsky* (FNC), es decir, con reglas de la forma $A -> B C$ o $A -> b$.

La tabla de análisis $t[i, j]$ almacena en cada celda los no terminales que derivan la subcadena comprendida entre las posiciones $i$ y $j$:
$ A in t[i, i+l] quad "sii" quad A dto(*) x_(i+1) dots x_(i+l) $

#figure(
  kind: "algorithm",
  supplement: [Algoritmo],
  pseudocode-list(
    booktabs: true,
    numbered-title: [Cocke-Kasami-Younger],
  )[
    - *Entrada:* $G = (Sigma, N, S, cal(P))$ en FNC y $x = x_1 dots x_T in Sigma^*$
    - *Salida:* tabla de análisis $t[i, j]$, $0 <= i < j <= T$
    + *para* $i = 0 dots T - 1$ *hacer*
      + $t[i, i+1] = t[i, i+1] union {A : (A -> b) in cal(P); thick b = x_(i+1)}$
    + *para* $l = 2 dots T$ *hacer*
      + *para* $i = 0 dots T - l$ *hacer*
        + *para* $k = 1 dots l - 1$ *hacer*
          + $t[i, i+l] = t[i, i+l] union {A : (A -> B C) in cal(P); thick B in t[i, i+k]; thick C in t[i+k, i+l]}$
    + *si* $S in t[0, T]$ *entonces* $x in L(G)$ *si no* $x in.not L(G)$
  ],
) <alg:cky>

El coste temporal es $O(|cal(P)| dot T^3)$: hay $O(T^2)$ celdas y en cada una se prueban $O(T)$ puntos de corte $k$.

==== Ejemplo

Sea la gramática en FNC con reglas $S -> A S | b$ y $A -> A S | A A | a$, y la cadena $x = a a b$ ($T = 3$). En la tabla de la @fig:cky_example cada entrada se numera y se anota como (no terminal, entrada de la izquierda, entrada de la derecha) para poder reconstruir los árboles; las entradas $(dot, 0, 0)$ provienen de reglas $A -> b$. Las filas indican $i$, las columnas $j$ y la diagonal $l = j - i$.

#figure(
  grid(
    columns: 2,
    column-gutter: 1cm,
    align: horizon,
    table(
      columns: (auto, 2.6cm, 2.6cm, 2.6cm),
      align: center + horizon,
      inset: (x: 5pt, y: 8pt),
      stroke: (x, y) => if x >= 1 and y >= 1 and x >= y { 0.6pt } else { none },
      [], [$j = 1$], [$j = 2$], [$j = 3$],
      [$i = 0$], [1: (A, 0, 0)], [4: (A, 1, 2)], text(size: 8.5pt)[7: (S, 1, 5) \ 8: (A, 1, 5) \ 9: (A, 1, 6) \ 10: (S, 4, 3) \ 11: (A, 4, 3)],
      [$i = 1$], [], [2: (A, 0, 0)], [5: (S, 2, 3) \ 6: (A, 2, 3)],
      [$i = 2$], [], [], [3: (S, 0, 0)],
    ),
    grid(
      columns: 2,
      column-gutter: 0.8cm,
      align: bottom,
      syntax-tree(spread: 0.8, ([$S_7$], ([$A_1$], term[a]), ([$S_5$], ([$A_2$], term[a]), ([$S_3$], term[b])))),
      syntax-tree(spread: 0.8, ([$S_10$], ([$A_4$], ([$A_1$], term[a]), ([$A_2$], term[a])), ([$S_3$], term[b]))),
    ),
  ),
  kind: image,
  caption: [Tabla CKY para la cadena $a a b$ y los dos árboles de análisis reconstruidos a partir de las entradas 7 y 10.],
) <fig:cky_example>

Como $S in t[0, 3]$ (entradas 7 y 10), $a a b in L(G)$, y además la cadena tiene dos análisis distintos.

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////

= Modelos para la predicción estructurada estadística <sec:modelos>

Contenidos de este bloque:
- Introducción: modelos estadísticos.
- Modelos generativos: _Probabilistic Finite-State Automata_ (PFSA) y _Probabilistic Context-Free Grammars_ (PCFG).
- _Probabilistic Graphical Models_ (PGM): _Bayesian Networks_ (modelos gráficos dirigidos) y _Markov Random Fields_ (modelos gráficos no dirigidos).
- Modelos discriminativos: _Conditional Random Fields_ (CRF).

== Introducción: modelos estadísticos <sec:modelos_estadisticos>

=== Incertidumbre

#pred-diagram(x: $bold(x) in cal(X)$, y: $bold(y) in Y(M)$, model: $M$)

En la vida real aparecen distintas fuentes de incertidumbre:
- *Fragilidad*: falta de robustez debida a una representación imperfecta de los objetos de entrada. _[Incertidumbre sobre la representación de las entradas.]_
- *Ambigüedad*: varias interpretaciones (hipótesis) son plausibles debido a la variabilidad y la dificultad de la tarea. _[Incertidumbre sobre las salidas.]_
- *Incompletitud*: modelos inadecuados por un conocimiento incompleto o insuficiente.

#note(title: "Solución estadística")[
  Basada en la *teoría estadística de la decisión*: $f : cal(X) -> Y(M)$ con
  $ hat(bold(y)) = f(bold(x)) = arg max_(bold(y) in Y(M)) P(bold(y) | bold(x)) $
]

=== Taxonomía de modelos

- *Modelos discriminativos* (probabilidad condicional $P(y | x)$):
  $ f(x) = arg max_(y in Y(M)) P(y | x) $
- *Modelos generativos* (probabilidad conjunta $P(x, y)$):
  $ f(x) = arg max_(y in Y(M)) P(x, y) = arg max_(y in Y(M)) P(x | y) dot P(y) $

El problema: no conocemos la distribución real $P$, por lo que para usar esa regla en la práctica hay que resolver tres problemas independientes:

+ *Modelo estadístico*: sustituye a los modelos reales desconocidos por una familia de distribuciones que dependen de unos parámetros $theta$:
  $ P(y | x) "o" P(x | y) "y" P(y) quad "vs." quad P_theta (y | x) "o" P_theta (x | y) "y" P_theta (y) $
  P. ej. funciones discriminantes, _Neural Networks_, _Gaussian Mixture Models_ (GMM), modelos con variables ocultas (HMM, _alignment models_), _log-linear models_...

+ *Regla de decisión (búsqueda o decodificación)*: Con el modelo ya fijado, dada una entrada $x$ hay que calcular el $arg max$, y aquí está el problema de la predicción estructurada: $Y(M)$ es enorme. Hay exponencialmente muchas secuencias de etiquetas o árboles posibles, así que no se pueden probar uno a uno. Hacen falta algoritmos eficientes:
  - programación dinámica: Viterbi para HMM, CKY para gramáticas; o
  - búsqueda heurística: A#super[\*], beam search…
  (En la práctica, para problemas complejos se usan algoritmos de programación dinámica con heurísticas para reducir el espacio de búsqueda y que la decodificación sea tratable, aunque se pierda la garantía de encontrar la solución óptima.)

+ *Criterio de entrenamiento*: Para aprender los parámetros desconocidos $theta$ a partir de datos de entrenamiento. El criterio es la función que se optimiza para elegir θ:
  - *máxima verosimilitud*: el θ que hace más probables los datos de entrenamiento (por ejemplo, contar frecuencias, o el algoritmo EM si hay variables ocultas);
  - *máximo a posteriori*: lo mismo, pero añadiendo un conocimiento previo sobre θ (una regularización);
  - *mínimo error de clasificación*: optimizar directamente la tasa de aciertos.

Los algoritmos concretos dependen del modelo estadístico elegido.

=== Análisis sintáctico estadístico

#pred-diagram(
  x: $x in cal(X)$,
  y: $t_x in cal(T)(G)$,
  box-label: [_Parsing_],
  model: $G_theta$,
)

$ t_x in cal(T)(G) : "yield"(t_x) = x quad "sii" quad S dto(+) x quad "sii" quad x in L(G) $

La solución estadística consiste en usar *modelos probabilísticos*:
$ x in L(G) quad ==> quad P(x ; G_theta) quad equiv quad P(x ; theta) quad equiv quad P_theta (x) $
donde $theta$ es el vector de parámetros del modelo probabilístico $G_theta$.

== Lenguajes probabilísticos <sec:lenguajes_prob>

- *Lenguaje*: $L subset.eq Sigma^*$, dado un alfabeto $Sigma$.
- *Lenguaje generado por una gramática* $G = (Sigma, N, S, cal(P))$: $L(G) = {x | x in Sigma^* : S dto(+) x} subset.eq Sigma^*$.

#note(title: "Definición: lenguaje probabilístico")[
  Un *lenguaje probabilístico* sobre un alfabeto $Sigma$ es un par $(L, phi)$ donde:
  - $L subset.eq Sigma^*$ es el lenguaje de siempre, el conjunto de cadenas válidas. Se llama *lenguaje característico*.
  - $phi : Sigma^* -> [0, 1]$ es una *función probabilística computable* (asigna a cada cadena posible su probabilidad) que cumple:
    + $x in.not L ==> phi(x) = 0, quad forall x in Sigma^*$: las cadenas que no pertenecen al lenguaje tienen probabilidad cero.
    + $x in L ==> 0 < phi(x) <= 1, quad forall x in Sigma^*$: toda cadena del lenguaje tiene probabilidad estrictamente positiva.
    + $sum_(x in L) phi(x) = 1$:  las probabilidades suman 1, así que $phi$ es una distribución de probabilidad de verdad.
]

*Ejemplo* @booth1973applying. Sea $Sigma = {a, b}$ y el lenguaje $L = {a^n b^n | n >= 0}$ con
$ phi(x) = 0 quad forall x in.not L, quad quad phi(a^n b^n) = 1 / (e n!) $
Es un lenguaje probabilístico, ya que
$ sum_(x in L) phi(x) = sum_(0 <= n < infinity) 1 / (e n!) = 1 / e sum_(0 <= n < infinity) 1 / n! = 1 / e e = 1 $

#note(title: [Teorema @wetherell1980probabilistic])[
  Sea $(L, phi)$ un lenguaje probabilístico infinito. Entonces, para todo $epsilon > 0$ existe un $n >= 0$ tal que
  $ |x| >= n ==> epsilon > phi(x) $
  Intuitivamente: como las probabilidades suman 1, solo un número finito de cadenas puede tener probabilidad mayor o igual que $epsilon$ (a lo sumo $1 \/ epsilon$); basta tomar $n$ mayor que la longitud de todas ellas. Las cadenas largas son necesariamente poco probables.
]

== Cobertura del curso <sec:cobertura>

#figure(
  table(
    columns: 3,
    align: (left, left, left),
    stroke: none,
    table.hline(stroke: 1pt),
    table.header([*Problema*], [*Tipo*], [*Modelos*]),
    table.hline(stroke: 0.5pt),
    table.cell(rowspan: 2)[No estructurado], [Generativo], [_Naive Bayes Classifier_],
    [Discriminativo], [_Logistic Regression_ \ _Perceptron_ \ _Support Vector Machines_ (SVM) \ _Neural Networks_],
    table.hline(stroke: 0.5pt),
    table.cell(rowspan: 2)[Estructurado], [Generativo], [*_Probabilistic Finite-State Automata_* \ *_Hidden Markov Models_ (HMM)* \ *_Probabilistic Context-Free Grammars_*],
    [Discriminativo], [*_Conditional Random Fields_ (CRF)* \ *_Structured Perceptron_* \ _Structured Support Vector Machines_ \ _Recurrent Neural Networks_ (_encoder-decoder_)],
    table.hline(stroke: 1pt),
  ),
  caption: [Modelos tratados en la asignatura.],
) <tab:cobertura>

== _Probabilistic Finite-State Automata_ (PFSA) <sec:pfsa>

#note(title: [Definición: _Probabilistic Finite-State Automaton_ (PFSA)])[
  Un *_Probabilistic Finite-State Automaton_* (PFSA) $A$ sobre un semianillo probabilístico es una tupla $A = (Sigma, Q, delta, I, F, P)$ donde:
  - $Sigma$ es el alfabeto;
  - $Q$ es un conjunto finito de estados;
  - $delta subset.eq Q times (Sigma union {epsilon}) times Q$ es el conjunto de transiciones;
  - $I : Q -> RR^+$ son las probabilidades de estado inicial;
  - $F : Q -> RR^+$ son las probabilidades de estado final;
  - $P : delta -> RR^+$ son las probabilidades de transición.

  $I$, $F$ y $P$ son funciones probabilísticas que deben cumplir:
  $
    sum_(q in Q) I(q) & = 1, \
    forall q in Q quad F(q) + sum_(a in Sigma; thin q' in Q) P(q, a, q') & = 1.
  $
]

Es decir, en cada estado la probabilidad de *terminar* más la de *salir* por alguna transición suma 1.

=== Ejemplo

En la @fig:pfsa, cada estado se etiqueta como _estado/probabilidad final_ $F(q)$ y cada transición como _símbolo/probabilidad_. Además $I(q_0) = 1$ y $I(q_i) = 0$ para todo $i != 0$. Se puede comprobar que, p. ej. en $q_0$: $F(q_0) + 0.3 + 0.2 + 0.05 + 0.4 = 0.05 + 0.95 = 1$.

#figure(
  automaton(
    (
      q0: (q0: "c", q1: "a", q2: "a"),
      q1: (q1: "c", q3: "b"),
      q2: none,
      q3: none,
    ),
    initial: "q0",
    final: ("q0", "q1", "q2", "q3"),
    labels: (
      q0: [0/.05], q1: [1/.2], q2: [2/1.], q3: [3/1.],
      q0-q0: [c/.3], q0-q1: [a/.2 \ d/.05], q0-q2: [a/.4],
      q1-q1: [c/.4], q1-q3: [b/.4],
    ),
    layout: (q0: (0, 0), q1: (3.4, 1.1), q2: (3.4, -1.3), q3: (6.6, 1.1)),
    style: automaton-style + (
      state: (radius: 0.62),
      q0: (initial: (label: none)),
      q0-q1: (curve: 0.3, label: (dist: 0.5)),
      q0-q2: (curve: -0.3, label: (dist: -0.35)),
      q1-q3: (curve: 0),
    ),
  ),
  caption: [_Probabilistic Finite-State Automaton_ (PFSA).],
) <fig:pfsa>

Probabilidad de la cadena $a c b$ a través del camino $q_0 q_1 q_1 q_3$:
$ P_A (a c b, q_0 q_1 q_1 q_3) = I(q_0) dot P(q_0, a, q_1) dot P(q_1, c, q_1) dot P(q_1, b, q_3) dot F(q_3) = 1 dot 0.2 dot 0.4 dot 0.4 dot 1 = 0.032 $

La cadena $a$ puede aceptarse por dos caminos distintos:
$
  P_A (a, q_0 q_1) & = I(q_0) dot P(q_0, a, q_1) dot F(q_1) = 0.04 \
  P_A (a, q_0 q_2) & = I(q_0) dot P(q_0, a, q_2) dot F(q_2) = 0.4 \
  P_A (a) & = P_A (a, q_0 q_1) + P_A (a, q_0 q_2) = 0.44
$

=== Probabilidad de una cadena

Una cadena $x in Sigma^*$ es *aceptada* por el PFSA $A$ si existe un *camino* $pi$ (secuencia de transiciones) por los estados $s_0 s_1 dots s_k$ ($|x| <= k$, debido a las posibles transiciones $epsilon$) tal que
$ pi = (s_0, x_1, s_1) dot (s_1, x_2, s_2) dots.c (s_(k-1), x_k, s_k) $

Denotamos por $Pi_A (x)$ el conjunto de caminos de $A$ que aceptan $x$.

- *Probabilidad de un camino*:
  $ P_A (x, pi) = I(s_0) dot (product_(i=1)^k P(s_(i-1), x_i, s_i)) dot F(s_k) $
- *Probabilidad de una cadena*:
  $ P_A (x) = sum_(pi in Pi_A (x)) P_A (x, pi) $
- *Probabilidad del mejor camino* (aproximación de Viterbi):
  $ hat(P)_A (x) = max_(pi in Pi_A (x)) P_A (x, pi) $
- *Mejor camino*:
  $ hat(pi)(x) = arg max_(pi in Pi_A (x)) P_A (x, pi) $
- *Lenguaje aceptado por un PFSA*:
  $ L(A) = {x in Sigma^* | P_A (x) > 0} $

=== Propiedades

#note(title: "Consistencia")[
  Un PFSA $A$ es *consistente* si y solo si
  $ sum_(x in L(A)) P_A (x) = 1 $
]

- *Definición*: un estado de un PFSA $A$ es *útil* si aparece en al menos un camino válido de $Pi_A$.
- *Proposición*: un PFSA es consistente si todos sus estados son útiles.

== _Hidden Markov Models_ (HMM) <sec:hmm>

#note(title: [Definición: _Hidden Markov Model_ (HMM)])[
  Un *_Hidden Markov Model_* (HMM) $M$ sobre un semianillo probabilístico es una tupla $M = (Sigma, Q, F, I, T, E)$ donde:
  - $Sigma$ es el alfabeto;
  - $Q$ es un conjunto finito de estados;
  - $q_f in F subset.eq Q$ es un estado especial (final);
  - $I : (Q - {q_f}) -> RR^+$ es la función de probabilidad de estado inicial;
  - $T : (Q - {q_f}) times Q -> RR^+$ es la función de probabilidad de transición entre estados;
  - $E : (Q - {q_f}) times Sigma -> RR^+$ es la función de probabilidad de emisión de símbolos (asociada a los estados).

  $I$, $T$ y $E$ son funciones probabilísticas que deben cumplir:
  $
    sum_(q in (Q - {q_f})) I(q) & = 1, \
    forall q in (Q - {q_f}) quad sum_(q' in Q) T(q, q') & = 1, \
    forall q in (Q - {q_f}) quad sum_(a in Sigma) E(q, a) & = 1.
  $
]

A diferencia del PFSA, en un HMM los símbolos se *emiten en los estados* (no en las transiciones), y la secuencia de estados está *oculta*: solo se observa la secuencia de símbolos emitidos.

=== Ejemplo

#figure(
  grid(
    columns: 1,
    row-gutter: 0.5em,
    align: center,
    // Tablas de emisión, alineadas con q1, q2, q3
    {
      let emis(pa, pb) = table(
        columns: 2,
        stroke: (x, y) => (right: 0.6pt + if x == 1 { black } else { black }),
        inset: (x: 5pt, y: 2pt),
        [a], [#pa], [b], [#pb],
      )
      grid(
        columns: 4, column-gutter: 1.1cm, align: bottom,
        emis(0.9, 0.1), emis(0.1, 0.9), emis(0.9, 0.1), box(width: 1cm),
      )
    },
    automaton(
      (q1: (q1: "", q2: "", q3: ""), q2: (q2: "", q3: ""), q3: (q3: "", q4: ""), q4: none),
      initial: "q1",
      final: none,
      labels: (
        q1-q1: $0.7$, q1-q2: $0.2$, q1-q3: $0.1$,
        q2-q2: $0.9$, q2-q3: $0.1$,
        q3-q3: $0.9$, q3-q4: $0.1$,
      ),
      layout: (q1: (0, 0), q2: (2.6, 0.9), q3: (5.2, 0), q4: (7.6, 0)),
      style: automaton-style + (
        state: (radius: 0.45),
        q1: (initial: (label: none)),
        q1-q2: (curve: 0), q2-q3: (curve: 0), q3-q4: (curve: 0),
        q1-q3: (curve: -0.6, label: (dist: -0.3)),
      ),
    ),
  ),
  kind: image,
  caption: [_Hidden Markov Model_ (HMM). Sobre los estados $q_1$, $q_2$ y $q_3$ se muestran las probabilidades de emisión de los símbolos $a$ y $b$; $q_4$ es el estado final $q_f$. Probabilidades iniciales: $I(q_1) = 1$; $I(q_2) = I(q_3) = I(q_4) = 0$.],
) <fig:hmm>

La cadena $a b a$ puede generarse a través de tres secuencias de estados:

#{
  set text(size: 9pt)
  $
    P_M (a b a, q_1 q_2 q_3 q_4) & = I(q_1) E(a, q_1) T(q_1, q_2) E(b, q_2) T(q_2, q_3) E(a, q_3) T(q_3, q_4) & = 1.458 dot 10^(-3) \
    P_M (a b a, q_1 q_1 q_3 q_4) & = I(q_1) E(a, q_1) T(q_1, q_1) E(b, q_1) T(q_1, q_3) E(a, q_3) T(q_3, q_4) & = 0.567 dot 10^(-3) \
    P_M (a b a, q_1 q_3 q_3 q_4) & = I(q_1) E(a, q_1) T(q_1, q_3) E(b, q_3) T(q_3, q_3) E(a, q_3) T(q_3, q_4) & = 0.729 dot 10^(-3) \
    P_M (a b a) & = & 2.754 dot 10^(-3)
  $
}

=== Probabilidad de una cadena

- La probabilidad de que el HMM $M$ acepte la cadena $x = x_1 dots x_k in Sigma^*$ a través del camino (secuencia de estados) $pi = s_1 dots s_k$ es:
  $ P_M (x, pi) = I(s_1) dot product_(i=2)^k T(s_(i-1), s_i) dot product_(i=1)^k E(x_i, s_i) $
- La probabilidad de que el HMM $M$ acepte la cadena $x in Sigma^*$ es:
  $ P_M (x) = sum_(pi in Pi_M (x)) P_M (x, pi) $

#note(title: "Observación")[
  En el ejemplo, el camino incluye además la transición final hacia el estado $q_f$ (que no emite ningún símbolo), por lo que aparece el factor adicional $T(s_k, q_f)$.
]

#note(title: "Equivalencia HMM–PFSA")[
  Dado un HMM $M$, existe un PFSA $A$ tal que $P_M (x) = P_A (x) quad forall x in Sigma^*$.
]

== _Probabilistic Context-Free Grammars_ (PCFG) <sec:pcfg>

#note(title: [Definición: _Probabilistic Context-Free Grammar_ (PCFG)])[
  Una *_Probabilistic Context-Free Grammar_* (PCFG) es un par $G_theta = (G, P)$ donde:
  - $G = (Sigma, N, S, cal(P))$ es la *gramática característica*;
  - $P : cal(P) -> ]0, 1]$ asigna probabilidades a las reglas:
    $ forall A_i in N quad P(A_i -> alpha_j) equiv P(r_(i j)) equiv P(A_i -> alpha_j | A_i) equiv P(alpha_j | A_i) $
    de modo que, siendo $n_i$ el número de reglas con $A_i$ en la parte izquierda,
    $ sum_(1 <= j <= n_i) P(alpha_j | A_i) = 1 $
]

=== Derivación probabilística

Dada una secuencia de eventos estocásticos (aplicaciones de reglas)
$ S = alpha_0 dto(r_1) alpha_1 dto(r_2) alpha_2 dots.c alpha_(m-1) dto(r_m) alpha_m = x $
la probabilidad de que $G_theta$ genere $x$ mediante la secuencia de reglas $t_x = r_1 dots.c r_m$ es
$ P_theta (x, t_x) = P(r_1) dot P(r_2 | r_1) dots.c P(r_m | r_1 dots.c r_(m-1)) $

- *Problema*: el cálculo de estas probabilidades condicionadas.
- *Restricción* (hipótesis de independencia, propia de las gramáticas _incontextuales_): $P(r_j | r_1 dots.c r_(j-1)) = P(r_j)$, de modo que
  $ P_theta (x, t_x) = product_(j = 1 dots.c m) P(r_j) $

=== Ejemplo

Se añaden probabilidades a la gramática de la @tab:cfg_example:

#figure(
  table(
    columns: 4,
    align: left,
    stroke: 0.5pt,
    inset: (x: 8pt, y: 4pt),
    [$S -> "NP" "VP" quad 1.0$], [$"VP" -> V "NP" quad 0.7$], [$V -> mono("saw") quad 1.0$], [$"NP" -> mono("saw") quad 0.04$],
    [$"NP" -> "NP" "PP" quad 0.4$], [$"VP" -> "VP" "PP" quad 0.3$], [$"NP" -> mono("astronomers") quad 0.1$], [$"NP" -> mono("stars") quad 0.18$],
    [$"PP" -> P "NP" quad 1.0$], [$P -> mono("with") quad 1.0$], [$"NP" -> mono("ears") quad 0.18$], [$"NP" -> mono("telescopes") quad 0.1$],
  ),
  caption: [Gramática incontextual probabilística.],
) <tab:pcfg_example>

Puede comprobarse que, para cada no terminal, las probabilidades de sus reglas suman 1 (p. ej. para NP: $0.4 + 0.1 + 0.18 + 0.04 + 0.18 + 0.1 = 1$).

La derivación del primer árbol tiene probabilidad:

#{
  set text(size: 9pt)
  $
    S & dto(1.0) "NP" "VP" dto(0.1) mono("astronomers") "VP" dto(0.7) mono("astronomers") V "NP" dto(1.0) mono("astronomers saw") "NP" dto(0.4) \
    & mono("astronomers saw") "NP" "PP" dto(0.18) mono("astronomers saw stars") "PP" dto(1.0) mono("astronomers saw stars") P "NP" dto(1.0) \
    & mono("astronomers saw stars with") "NP" dto(0.1) mono("astronomers saw stars with telescopes") quad = 0.504 dot 10^(-3)
  $
}

#figure(
  grid(
    columns: 2,
    column-gutter: 1.5cm,
    align: bottom,
    [$0.504 dot 10^(-3)$ #v(-0.5em)
      #syntax-tree(spread: 0.7,
        ([S], ([NP], [astronomers]), ([VP], ([V], [saw]), ([NP], ([NP], [stars]), ([PP], ([P], [with]), ([NP], [telescopes])))))
      )],
    [$0.378 dot 10^(-3)$ #v(-0.5em)
      #syntax-tree(spread: 0.7,
        ([S], ([NP], [astronomers]), ([VP], ([VP], ([V], [saw]), ([NP], [stars])), ([PP], ([P], [with]), ([NP], [telescopes]))))
      )],
  ),
  caption: [Probabilidades de los dos árboles de análisis de _astronomers saw stars with telescopes_. El primero es el árbol más probable.],
) <fig:pcfg_trees>

La probabilidad de la cadena es la suma de ambas: $P_theta (x) = 0.504 dot 10^(-3) + 0.378 dot 10^(-3) = 0.882 dot 10^(-3)$.

=== Probabilidades asociadas

Sea $cal(T)_x$ el conjunto de árboles de análisis de $x$.

- *Probabilidad de un árbol de análisis*:
  $ P_theta (x, t_x) = product_(j = 1 dots.c m) P(r_j) $
- *Probabilidad de una cadena*:
  $ P_theta (x) = sum_(t_x in cal(T)_x) P_theta (x, t_x) $
- *Probabilidad del mejor árbol*:
  $ hat(P)_theta (x) = max_(t_x in cal(T)_x) P_theta (x, t_x) $
- *Lenguaje generado por una gramática probabilística*:
  $ L(G_theta) = {x in L(G) | P_theta (x) > 0} $

=== Propiedades

#note(title: "Consistencia")[
  Una gramática probabilística $G_theta = (G, P)$ es *consistente* si y solo si
  $ sum_(x in L(G)) P_theta (x) = 1 $
]

#note(title: [Teorema @booth1973applying])[
  Existen lenguajes probabilísticos $(L, phi)$ que *no pueden ser generados* por ninguna gramática probabilística $G_theta = (G, P)$.

  *Ejemplo*: sea el lenguaje probabilístico $L = {a^n b^n | n >= 0}$ con $phi(a^n b^n) = 1 \/ (e n!)$. No existe ninguna $G_theta$ tal que $phi(x) = P_theta (x) quad forall x in L$.
]

== _Probabilistic Graphical Models_ (PGM) <sec:pgm>

Los *_Probabilistic Graphical Models_* (PGM) son un marco que combina la *incertidumbre* (probabilidades) con la *estructura lógica* (restricciones de independencia) para hacer suposiciones simplificadoras sobre qué variables afectan a qué otras variables. Estas suposiciones pueden representarse a menudo de forma gráfica, dando lugar a la familia de modelos conocida como *modelos gráficos* (GM).

¿Por qué necesitamos modelos gráficos?
- Son la base del enfoque probabilístico de los _sistemas inteligentes_.
- Son una representación compacta de distribuciones de probabilidad conjuntas mediante grafos: una combinación perfecta entre la *teoría de la probabilidad* y la *teoría de grafos*.
- Permiten abstraer las relaciones de independencia condicional entre variables de los detalles de su forma paramétrica.
- Generalizan, entre otros, las _Neural Networks_ y los _Hidden Markov Models_ (HMM).

#figure(
  grid(
    columns: 2,
    column-gutter: 2cm,
    graph-diagram(
      ("1": (0, 0), "2": (-0.7, -1.2), "3": (0.7, -1.2), "4": (0, -2.4), "5": (1.4, -2.4)),
      (("1", "2"), ("1", "3"), ("2", "4"), ("3", "4"), ("3", "5")),
      directed: true,
    ),
    graph-diagram(
      ("1": (0, 0), "2": (-0.7, -1.2), "3": (0.7, -1.2), "4": (0, -2.4), "5": (1.4, -2.4)),
      (("1", "2"), ("1", "3"), ("2", "3"), ("2", "4"), ("3", "4"), ("3", "5")),
    ),
  ),
  caption: [Modelo gráfico dirigido (izquierda) y no dirigido (derecha) @murphy2012machine.],
) <fig:pgm_example>

Los *nodos* representan variables aleatorias y el *grafo* representa un conjunto de independencias y factoriza una distribución. Según el tipo de grafo distinguimos:
- *_Bayesian Networks_*: basadas en *grafos dirigidos*.
- *_Markov Random Fields_*: basados en *grafos no dirigidos*.

Los dos problemas clave son:
- *Inferencia*: deducir distribuciones de probabilidad a partir de otras dadas.
- *Aprendizaje*: obtener el modelo probabilístico a partir de observaciones.

Aplicaciones: diagnóstico médico, detección de fallos; *visión por computador* (segmentación de imágenes, reconstrucción 3D, análisis de escenas); *procesamiento del lenguaje natural* (reconocimiento del habla, extracción de información, traducción automática); *robótica* (planificación, detección)...

=== _Bayesian Networks_ (BN) <sec:bn>

#note(title: [Definición: _Bayesian Network_ (BN)])[
  Una *_Bayesian Network_* (BN) es un modelo gráfico que representa un conjunto de variables aleatorias y sus dependencias condicionales mediante un *grafo dirigido acíclico* (_Directed Acyclic Graph_, DAG).
  - Los *nodos* representan variables aleatorias (discretas o continuas).
  - Las *aristas* representan dependencias estadísticas entre las variables.
]

Por la regla de la cadena, cualquier distribución conjunta puede factorizarse. Hay dos casos extremos (para variables binarias):
+ Sin ninguna independencia:
  $ p(x_1, x_2, dots, x_T) = p(x_1) dot p(x_2 | x_1) dots p(x_T | x_1, dots, x_(T-1)) = p(x_1) product_(t=2)^T p(x_t | x_1, dots, x_(t-1)) $
  que requiere $2^T - 1$ parámetros.
+ Todas las variables independientes:
  $ p(x_1, x_2, dots, x_T) = product_(t=1)^T p(x_t) $
  que requiere $T$ parámetros.

Si no todas las dependencias son posibles, la factorización completa queda reflejada en el grafo (DAG) asociado a la red. Una BN con nodos $x_1, dots, x_T$ define la distribución conjunta
$ p(x_1, x_2, dots, x_T) = product_(t=1)^T p(x_t | phi(x_t)) $
donde $phi(x_t)$ denota las dependencias (los *padres*) del nodo $x_t$.

#figure(
  graph-diagram(
    ("A": (0, 0.6), "B": (0, -0.8), "C": (2, 0.5), "D": (4, -0.8), "E": (6, 0)),
    (("A", "C"), ("B", "C"), ("C", "E"), ("C", "D"), ("B", "D"), ("D", "E")),
    directed: true,
    radius: 0.4,
  ),
  caption: [Ejemplo de _Bayesian Network_.],
) <fig:bn_example>

Para la red de la @fig:bn_example:
$ p(A, B, C, D, E) = p(A) dot p(B) dot p(C | A, B) dot p(D | B, C) dot p(E | C, D) $

==== Inferencia

El objetivo es estimar la *probabilidad a posteriori* de alguna variable $x$ a partir de la distribución conjunta asociada a la BN, dada cierta evidencia $y$ (independientemente de los valores del resto de variables $z$):
$ P(x | y) = P(x, y) / P(y) quad "con" quad P(x, y) = sum_z P(x, y, z); quad P(y) = sum_(x, z) P(x, y, z) $

El reto es calcular *eficientemente* $P(x, y)$ y $P(y)$. La probabilidad a posteriori es útil para:
- *Predicción*: ¿cuál es la probabilidad de observar un síntoma sabiendo que el paciente tiene una determinada enfermedad?
- *Diagnóstico*: ¿cuál es la probabilidad de que una enfermedad sea el diagnóstico correcto dados ciertos síntomas?

==== Ejemplo detallado

Sean las variables *lluvia* $R in {n, y}$ (no llueve / sí llueve), *aspersor* $S in {s, r}$ (parado / funcionando) y *césped* $G in {d, w}$ (seco / mojado), con la red de la @fig:bn_sprinkler.

#figure(
  grid(
    columns: 4,
    column-gutter: 0.9cm,
    align: horizon,
    graph-diagram(
      ("R": (0, 0), "S": (1.5, 0.9), "G": (3, 0)),
      (("R", "S"), ("S", "G"), ("R", "G")),
      directed: true,
      radius: 0.35,
    ),
    table(
      columns: 2, stroke: none, align: center,
      table.header(table.cell(colspan: 2)[$P(R)$]),
      table.hline(stroke: 0.5pt),
      [$n$], [$y$],
      [0.8], [0.2],
    ),
    table(
      columns: 3, stroke: none, align: center,
      table.header(table.cell(colspan: 3)[$P(S | R)$]),
      table.hline(stroke: 0.5pt),
      [$R$], table.vline(stroke: 0.5pt), [$s$], [$r$],
      [$n$], [0.60], [0.40],
      [$y$], [0.99], [0.01],
    ),
    table(
      columns: 4, stroke: none, align: center,
      table.header(table.cell(colspan: 4)[$P(G | R, S)$]),
      table.hline(stroke: 0.5pt),
      [$R$], [$S$], table.vline(stroke: 0.5pt), [$d$], [$w$],
      [$n$], [$s$], [0.99], [0.01],
      [$n$], [$r$], [0.10], [0.90],
      [$y$], [$s$], [0.20], [0.80],
      [$y$], [$r$], [0.01], [0.99],
    ),
  ),
  kind: image,
  caption: [Red bayesiana lluvia ($R$) – aspersor ($S$) – césped ($G$) y sus tablas de probabilidad condicional.],
) <fig:bn_sprinkler>

La distribución conjunta es $P(R, S, G) = P(R) thin P(S | R) thin P(G | R, S)$.

#note(title: "Ejercicio: ¿probabilidad de que el aspersor funcione si no llueve y el césped está mojado?")[
  Se pide $P(S = r | R = n, G = w)$:
  $
    P(S = r | R = n, G = w) & = (P(R = n) P(S = r | n) P(w | n, r)) / (sum_(s' in {s, r}) P(R = n) P(s' | n) P(w | n, s')) \
    & = (0.8 dot 0.4 dot 0.9) / (0.8 dot 0.6 dot 0.01 + 0.8 dot 0.4 dot 0.9) = 0.288 / (0.0048 + 0.288) approx 0.984
  $
  _(Resolución propia; no aparece en las transparencias.)_
]

==== Independencia condicional

El conocimiento se representa mediante la noción de *independencia condicional*:
- Dos sucesos $a$ y $b$ son *independientes (incondicionalmente)*, $(a indep b)$, si
  $ p(a, b) = p(a) dot p(b) quad "o" quad p(a | b) = p(a) $
- Dos sucesos $a$ y $b$ son *condicionalmente independientes* dado $c$, $(a indep b | c)$, si
  $ p(a | b, c) = p(a | c) quad "o" quad p(a, b | c) = p(a | c) dot p(b | c) $

En una BN hay tres estructuras básicas, según dónde esté el nodo observado $c$ (sombreado):

#figure(
  table(
    columns: (auto, 1fr),
    stroke: none,
    align: (center + horizon, left + horizon),
    row-gutter: 0.6em,
    graph-diagram(
      ("a": (0, 0), "c": (1.5, 0), "b": (3, 0)),
      (("a", "c"), ("c", "b")),
      directed: true, shaded: ("c",),
    ),
    [
      *Dirección causal*: $(a indep b | c)$, pero $(a nindep b)$
      $ P(a, b | c) = (P(a) P(c | a) P(b | c)) / P(c) = P(a | c) P(b | c) quad "pero" quad P(a, b) != P(a) P(b) $
    ],
    graph-diagram(
      ("c": (0.75, 0.8), "a": (0, -0.3), "b": (1.5, -0.3)),
      (("c", "a"), ("c", "b")),
      directed: true, shaded: ("c",),
    ),
    [
      *Padre común*: $(a indep b | c)$, pero $(a nindep b)$
      $ P(a, b | c) = (P(c) P(a | c) P(b | c)) / P(c) = P(a | c) P(b | c) quad "pero" quad P(a, b) != P(a) P(b) $
    ],
    graph-diagram(
      ("a": (0, 0.8), "b": (1.5, 0.8), "c": (0.75, -0.3)),
      (("a", "c"), ("b", "c")),
      directed: true, shaded: ("c",),
    ),
    [
      *Estructura en V* (_v-structure_): $(a nindep b | c)$, pero $(a indep b)$
      $ P(a, b | c) != P(a | c) P(b | c) quad "pero" quad P(a, b) = sum_c P(a) P(b) P(c | a, b) = P(a) P(b) $
    ],
  ),
  kind: image,
  caption: [Reglas de independencia condicional e incondicional en _Bayesian Networks_.],
) <fig:bn_independence>

#note(title: "Intuición")[
  En las dos primeras estructuras, observar $c$ *bloquea* el flujo de información entre $a$ y $b$. En la estructura en V ocurre lo contrario: $a$ y $b$ son independientes a priori, pero observar el efecto común $c$ los hace dependientes (fenómeno de _explaining away_).
]

==== Ejemplos de modelos generativos como _Bayesian Networks_

*Clasificadores: _Naive Bayes_.* Se predice una única clase $y$ dado un vector de características $bold(x) = {x_1, x_2, dots, x_T}$, suponiendo que, *conocida la clase, todas las características son independientes*:
$ p(x_1, x_2, dots, x_T, y) = p(y) product_(t=1)^T p(x_t | y) $

*Predicción secuencial: modelo de Markov.*
$ p(x_1, x_2, dots, x_T) = p(x_1) dot p(x_2 | x_1) dot p(x_3 | x_2) dots = p(x_1) product_(t=2)^T p(x_t | x_(t-1)) $

*Etiquetado de secuencias: _Hidden Markov Model_ (HMM) de bigramas.*
$ P(x_1^T, y_1^T) = product_(t=1)^T q(y_t | y_(t-1)) dot e(x_t | y_t) $

#figure(
  grid(
    columns: 3,
    column-gutter: 1cm,
    align: bottom,
    graph-diagram(
      ("Y": (1.5, 1.1), "X1": (0, 0), "X2": (1, 0), "X3": (2, 0), "X4": (3, 0)),
      (("Y", "X1"), ("Y", "X2"), ("Y", "X3"), ("Y", "X4")),
      directed: true, shaded: ("X1", "X2", "X3", "X4"),
      labels: ("Y": $Y$, "X1": $X_1$, "X2": $X_2$, "X3": $X_3$, "X4": $X_4$),
      radius: 0.36,
    ),
    graph-diagram(
      ("X1": (0, 0), "X2": (1.2, 0), "X3": (2.4, 0), "d": (3.5, 0)),
      (("X1", "X2"), ("X2", "X3"), ("X3", "d")),
      directed: true, shaded: ("X1", "X2", "X3"), plain: ("d",),
      labels: ("X1": $X_1$, "X2": $X_2$, "X3": $X_3$, "d": $dots$),
      radius: 0.36,
    ),
    graph-diagram(
      ("Y1": (0, 1.2), "Y2": (1.2, 1.2), "d": (2.3, 1.2), "YT": (3.4, 1.2),
       "X1": (0, 0), "X2": (1.2, 0), "XT": (3.4, 0)),
      (("Y1", "Y2"), ("Y2", "d"), ("d", "YT"), ("Y1", "X1"), ("Y2", "X2"), ("YT", "XT")),
      directed: true, shaded: ("X1", "X2", "XT"), plain: ("d",),
      labels: ("Y1": $Y_1$, "Y2": $Y_2$, "YT": $Y_T$, "X1": $X_1$, "X2": $X_2$, "XT": $X_T$, "d": $dots$),
      radius: 0.36,
    ),
  ),
  caption: [_Bayesian Networks_ de _Naive Bayes_ (izquierda), _Markov Model_ (centro) y _Hidden Markov Model_ (derecha). Los nodos sombreados son observados.],
) <fig:bn_generative>

*Análisis sintáctico: _Probabilistic Context-Free Grammars_ (PCFG).* Un árbol de análisis también puede verse como una _Bayesian Network_: cada no terminal $Y_i$ es una variable y cada terminal $X_l$ una variable observada.

#figure(
  {
    let y(s, i) = [#s #text(fill: color-palette.secondary, size: 8pt)[$(Y_#i)$]]
    let x(i, w) = [#text(fill: color-palette.secondary, size: 8pt)[$(X_#i)$] #term(w)]
    syntax-tree(spread: 1.1,
      (y("S", 1),
        (y("NP", 2), (y("DT", 4), x(1, "The")), (y("NN", 5), x(2, "bear"))),
        (y("VP", 3), (y("VBD", 6), x(3, "ate")), (y("NP", 7), (y("DT", 8), x(4, "the")), (y("NN", 9), x(5, "fish")))),
      )
    )
  },
  caption: [Árbol de análisis visto como _Bayesian Network_.],
) <fig:bn_parsing>

$
  t_x = [S = alpha_0 dto(r_1) alpha_1 dto(r_2) alpha_2 dots.c alpha_(m-1) dto(r_m) alpha_m = bold(x)] quad quad P(x_1^T, t_x) = product_(i = 1 dots.c m) P(r_i)
$
$ P(r_i) = P(Y_j thin Y_k | Y_i) quad "o" quad P(r_i) = P(X_l | Y_i) $

== _Markov Random Fields_ (MRF) <sec:mrf>

Las _Bayesian Networks_ no pueden representar perfectamente todas las distribuciones. Los *_Markov Random Fields_* (MRF) son modelos gráficos basados en *grafos no dirigidos*:
- Los *nodos* representan las variables aleatorias.
- Las *aristas* representan algún tipo de interacción probabilística entre nodos vecinos: indican qué variables dependen entre sí.

#figure(
  undirected-graph(
    (
      "1": (0, 0), "2": (1.8, 0.9), "3": (1.8, -0.9), "5": (3.6, 0),
      "4": (5.4, 0.9), "6": (5.4, -0.9), "7": (7.2, 0),
    ),
    (
      ("1", "2"), ("1", "3"), ("2", "3"), ("2", "4"), ("2", "5"), ("3", "5"), ("3", "6"),
      ("4", "5"), ("4", "6"), ("4", "7"), ("5", "6"), ("5", "7"), ("6", "7"),
    ),
  ),
  caption: [Grafo no dirigido de ejemplo @murphy2012machine, usado en el resto de la sección.],
) <fig:mrf_graph>

=== Propiedades de independencia condicional

#note(title: "Propiedad de Markov global")[
  Los MRF definen relaciones de independencia condicional mediante simple *separación en el grafo*: para conjuntos de nodos $A$, $B$ y $C$,
  $ A perp B | C quad <==> quad C "separa" A "de" B "en el grafo" $
  es decir, todo camino entre un nodo de $A$ y uno de $B$ pasa por algún nodo de $C$.
]

En el grafo de la @fig:mrf_graph, ${1, 2} perp {6, 7} | {3, 4, 5}$.

=== Cliques

#note(title: "Definiciones")[
  - Un *clique* de un grafo no dirigido es un subgrafo completamente conexo.
  - Un *clique maximal* es un clique que no es subgrafo de ningún otro clique.
]

Para el grafo de la @fig:mrf_graph:
- *Cliques*: ${1, 2}; dots; {1, 2, 3}; {2, 3, 5}; {2, 4, 5}; {3, 5, 6}; {4, 5, 6}; dots; {4, 5, 6, 7}$.
- *No cliques*: ${1, 2, 3, 5}; {2, 3, 4, 5, 6}; dots$
- *Cliques maximales*: ${1, 2, 3}; {2, 3, 5}; {2, 4, 5}; {3, 5, 6}; {4, 5, 6, 7}$.

=== Factorización

#note(title: "Definición: factorización sobre un grafo")[
  Sea $G$ el grafo no dirigido subyacente a un MRF. Una distribución de probabilidad $P_G$ define una *factorización* sobre $G$ si se puede escribir como
  $ P_G (X_1, dots, X_T) = 1 / Z product_(C in Q) psi_C (V_C) $
  donde:
  - $V = {X_1, X_2, dots, X_T}$ es el conjunto de variables aleatorias;
  - $Q$ es el conjunto de todos los cliques (maximales) de $G$;
  - $V_C$ es el subconjunto de variables del clique $C$;
  - $psi_C$ es la *función de potencial* del clique $C$, que toma valores en $RR^(>0)$;
  - $Z$ es un factor (constante) de normalización, la *función de partición*:
    $ Z = sum_(X_1, dots, X_T) product_(C in Q) psi_C (V_C) $
]

*Ejemplo*. Para el grafo de la @fig:mrf_graph, $Q = {{1, 2, 3}; {2, 3, 5}; {2, 4, 5}; {3, 5, 6}; {4, 5, 6, 7}}$ y
$ P_G (1, 2, 3, 4, 5, 6, 7) = 1 / Z psi_1 (1, 2, 3) dot psi_2 (2, 3, 5) dot psi_3 (2, 4, 5) dot psi_4 (3, 5, 6) dot psi_5 (4, 5, 6, 7) $

#note(title: "MRF frente a BN")[
  Los MRF son más potentes que las _Bayesian Networks_, pero computacionalmente más difíciles de tratar (p. ej. por el cálculo de $Z$, que suma sobre todas las configuraciones de las variables).
]

==== Funciones de energía

A menudo conviene especificar las funciones de potencial mediante transformaciones exponenciales:
$
  P_G (X_1, dots, X_T) & = 1 / Z product_(C in Q) psi_C (V_C) \
  & = 1 / Z product_(C in Q) exp(-E_C (V_C)) \
  & = 1 / Z exp(- sum_(C in Q) E_C (V_C))
$
donde $E_C$ es la *función de energía* del clique $C$ (con valores en $RR$). Hay tipos de funciones de energía que pueden definirse mediante *funciones lineales generalizadas*:
$ E_C (V_C) = - sum_k theta_(C, k) thin f_(C, k) (V_C) $

== _Conditional Random Fields_ (CRF) <sec:crf>

Dadas una secuencia de entrada $x = x_1, x_2, dots, x_T in cal(X)^*$ y una de salida $y = y_1, y_2, dots, y_T in cal(Y)^*$, los *_Conditional Random Fields_* (CRF) @lafferty2001conditional son un *modelo discriminativo* definido por:
- unos *parámetros* $theta$;
- una *función de compatibilidad* $phi(x, y; theta) -> RR$, que asigna puntuaciones positivas altas a los pares $(x, y)$ compatibles;
- la *distribución condicional*:
  $ p(y | x; theta) = exp{phi(x, y; theta)} / Z(x; theta) quad "donde" quad Z(x; theta) = sum_(y' in cal(Y)^*) exp{phi(x, y'; theta)} $

=== Ingeniería de características

Las funciones de compatibilidad se construyen a partir de *características indicadoras* (de bigramas).

*Ejemplo: reconocimiento de entidades nombradas* (_Named-Entity Recognition_) @sutton2012introduction. Las entidades son personas (P), organizaciones (O), lugares (L) y otras (M). Con el etiquetado del conjunto de datos CoNLL 2003, el conjunto de etiquetas es $cal(Y) = {"B-P", "I-P", "B-O", "I-O", "B-L", "I-L", "B-M", "I-M", "O"}$, donde B- marca el comienzo de una entidad, I- su continuación y O una palabra fuera de cualquier entidad.

#figure(
  table(
    columns: 6,
    stroke: none,
    align: center,
    table.hline(stroke: 1pt),
    [U.N.], [official], [Ekeus], [heads], [for], [Baghdad],
    table.hline(stroke: 0.5pt),
    [B-O], [O], [B-P], [O], [O], [B-L],
    table.hline(stroke: 1pt),
  ),
  caption: [Frase etiquetada para reconocimiento de entidades nombradas.],
) <tab:ner>

- *Características etiqueta–etiqueta*:
  $ f_(i j)^(L L) (y_(t-1), y_t, x_t) = cases(1 & "si" y_(t-1) = i "y" y_t = j, 0 & "en otro caso") quad forall i, j in cal(Y) $
  Como hay 9 etiquetas distintas, hay $9^2 = 81$ características etiqueta–etiqueta.
- *Características etiqueta–palabra*:
  $ f_(i v)^(L W) (y_(t-1), y_t, x_t) = cases(1 & "si" y_t = i "y" x_t = v, 0 & "en otro caso") quad forall i in cal(Y), forall v in cal(V) $
  donde $cal(V)$ es el conjunto de todas las palabras del corpus. En CoNLL 2003, $|cal(V)| = 21 thin 249$ palabras, luego hay $9 dot 21 thin 249 = 191 thin 241$ características etiqueta–palabra.
- *Características etiqueta–observación*:
  $ f_(i b)^(L O) (y_(t-1), y_t, x_t) = cases(1 & "si" y_t = i "y" g_b (x_t), 0 & "en otro caso") quad forall i in cal(Y) $
  donde $g_b$ es un predicado sobre la observación (p. ej. «empieza por mayúscula»).

Todas estas funciones son binarias y se definen heurísticamente según la tarea.

=== Funciones de compatibilidad factorizadas

Sea $f(y_(t-1), y_t, x_t) = (f_1 (y_(t-1), y_t, x_t), dots, f_K (y_(t-1), y_t, x_t))$ un vector de características y $theta in RR^K$ un vector de pesos. Entonces
$
  phi(x, y; theta) = sum_(t=1)^T theta dot f(y_(t-1), y_t, x_t) = sum_(t=1)^T sum_(k=1)^K theta_k f_k (y_(t-1), y_t, x_t)
$

- Se define $y_0 = "NULL"$ para que el bigrama $y_0 y_1$ esté definido en $t = 1$.
- Esta factorización permite algoritmos eficientes, ya que si $y != y'$ comparten bigramas, comparten también sus puntuaciones.

Sustituyendo, el modelo queda
$
  p(y | x; theta) = exp{sum_(t=1)^T sum_(k=1)^K theta_k f_k (y_(t-1), y_t, x_t)} / Z(x; theta), quad Z(x; theta) = sum_(y' in cal(Y)^*) exp{sum_(t=1)^T sum_(k=1)^K theta_k f_k (y'_(t-1), y'_t, x_t)}
$

- Las características $f(dot)$ están dadas (dependen del problema).
- $theta in RR^K$ son los parámetros del modelo.
- Los CRF son *_log-linear models_* sobre las funciones de características.

=== Los tres problemas de los CRF

+ *Inferencia*: calcular la probabilidad de una secuencia de salida $y$ para $x$:
  $ p(y | x; theta) = 1 / Z(x; theta) product_(t=1)^T exp{sum_(k=1)^K theta_k f_k (y_(t-1), y_t, x_t)} $
+ *Decodificación*: predecir la mejor secuencia de salida para $x$:
  $ arg max_(y in cal(Y)^*) p(y | x; theta) = arg max_y product_(t=1)^T exp{sum_(k=1)^K theta_k f_k (y_(t-1), y_t, x_t)} $
  ($Z(x; theta)$ no depende de $y$, por lo que no interviene en la maximización.)
+ *Estimación de parámetros*: aprender $theta$ a partir de datos de entrenamiento
  $ {(x^((1)), y^((1))), (x^((2)), y^((2))), dots, (x^((N)), y^((N)))} $

#note(title: "Bibliografía de esta parte")[
  - _Probabilistic Context-Free Grammars_ (PCFG): @booth1973applying, @wetherell1980probabilistic, @collins2011pcfg.
  - _Probabilistic Graphical Models_ (PGM): @bishop2006pattern (cap. 8), @koller2007graphical, @murphy2012machine.
  - _Conditional Random Fields_ (CRF): @lafferty2001conditional, @sutton2007introduction, @sutton2012introduction.
]

#pagebreak()

#bibliography("bibliography.bib", title: "Referencias")
