// Shared style for the project documents (adapted from example.typ).
// Usage:  #import "template.typ": *
//         #show: project.with(title: [...], subtitle: [...])

#import "@preview/cetz:0.4.2"

#let color-palette = (
  primary: rgb("#164863"),
  secondary: rgb("#2A6A8C"),
  accent: rgb("#F7693B"),
  text: rgb("#111111"),
  background: rgb("FFFFFF"),
  highlight: rgb("#FFE964"),
)

// Numbers for the documents, produced by `uv run python docs/make_figures.py`.
#let m = json("figures/metrics.json")

#let num(x, d: 1) = {
  let s = str(calc.round(x, digits: d))
  if d > 0 and not s.contains(".") { s += "." + "0" * d }
  s
}
#let pct(x, d: 1) = num(x * 100, d: d) + "%"
#let thousands(n) = {
  let s = str(n)
  if s.len() > 3 { s.slice(0, s.len() - 3) + "," + s.slice(s.len() - 3) } else { s }
}
#let MUL = "multiplicative (log)"
#let ADD = "additive (kcal)"
#let size-row(v, i) = m.by_size.filter(r => r.variant == v and r.order == i).first()

#let note(title: none, body) = {
  v(5pt)
  block(
    width: 100%,
    breakable: false,
    inset: 0.3em,
    outset: 0.3em,
    fill: color-palette.background.lighten(-7%),
    stroke: (left: (thickness: 5pt, paint: color-palette.primary, dash: "solid")),
  )[
    #set par(leading: 5.6pt, spacing: 10pt)
    #pad(
      left: 0.3em,
      right: 0.3em,
      strong(text(size: 9pt, smallcaps(title))) + block(above: 0.8em, text(size: 9pt, body)),
    )
  ]
  v(5pt)
}

// Row of boxes joined by arrows.
#let pipeline(steps, w: 2.55, h: 1.0, gap: 0.55) = align(center, block(above: 1em, below: 1.2em, cetz.canvas({
  import cetz.draw: *
  let arrow = (mark: (end: "stealth", fill: black, scale: 0.8), stroke: 0.6pt)
  for (i, s) in steps.enumerate() {
    let x = i * (w + gap)
    rect((x, 0), (x + w, h), stroke: 0.6pt + color-palette.primary, radius: 3pt, name: "b" + str(i))
    content("b" + str(i), align(center, text(size: 8.5pt, s)))
    if i > 0 { line((x - gap, h / 2), (x, h / 2), ..arrow) }
  }
})))

#let project(title: none, subtitle: none, eyebrow: [Conformal Calorie Estimation], author: "David González Durán", date: none, body) = {
  set document(title: title)
  set text(font: "Source Sans 3", size: 10pt, fill: color-palette.text, number-type: "lining", lang: "en")
  set par(justify: true, leading: 6pt, spacing: 12pt)

  show link: it => underline(stroke: color-palette.accent, offset: 1.2pt, text(fill: color-palette.accent, weight: "medium", it))
  show ref: set text(weight: 600, fill: color-palette.secondary)
  show footnote: set text(weight: 600, fill: color-palette.secondary)

  // Plain numbers: references add "Section" themselves, and the chapter
  // opener below prints "Section N" above the title.
  set heading(numbering: "1.")
  show heading: set block(below: 12pt)
  show heading: smallcaps
  show heading.where(level: 1): set text(18pt, weight: "semibold", number-type: "old-style")
  show heading.where(level: 2): set text(14pt, weight: "semibold", number-type: "old-style")
  show heading.where(level: 3): set text(12pt, weight: "semibold", number-type: "old-style")
  show heading.where(level: 1): it => {
    if it.numbering == none or not it.outlined { return it }
    pagebreak(weak: true)
    v(1.5cm)
    block(below: 0.6em, text(13pt, weight: "semibold", fill: color-palette.secondary, number-type: "lining")[
      #smallcaps[Section #counter(heading).display("1")]
    ])
    block(below: 0.8em, text(22pt, weight: "semibold", hyphenate: false)[#smallcaps(it.body)])
    line(length: 100%, stroke: 0.9pt + color-palette.primary)
    v(0.8cm)
  }

  set highlight(fill: color-palette.highlight, radius: 5pt, extent: 0.2em)
  show strong: set text(color-palette.text)
  set list(marker: ([•], [--], []), indent: 6pt)
  set enum(indent: 6pt)
  show raw.where(block: false): set text(size: 9pt)
  show raw.where(block: true): it => block(width: 100%, inset: 8pt, radius: 3pt, fill: luma(245), text(size: 8.5pt, it))
  set table(stroke: (x, y) => (top: if y <= 1 { 0.6pt } else { 0.3pt + luma(200) }, bottom: 0.6pt))
  show table.cell.where(y: 0): set text(weight: 600)
  show figure.caption: it => align(left)[
    #text(size: 10pt, weight: 600)[#it.supplement #context it.counter.display(it.numbering):]
    #text(size: 10pt)[#it.body]
  ]

  // ------------------------------------------------------------------ cover
  set page(paper: "a4", margin: (left: 2cm, right: 2cm, top: 2.5cm, bottom: 2cm), numbering: none)
  align(center, block(width: 100%, height: 100%)[
    #v(3.5cm)
    #text(size: 19pt, weight: 500)[#smallcaps(eyebrow)]
    #v(5pt)
    #text(size: 15pt)[#smallcaps[Nutrition5k · Vision Transformers · Conformal prediction]]
    #v(30pt)
    #line(length: 100%, stroke: 1.5pt)
    #v(5pt)
    #text(size: 26pt, weight: "bold")[#smallcaps(title)]
    #v(5pt)
    #text(size: 20pt, weight: "medium")[#smallcaps(subtitle)]
    #v(5pt)
    #line(length: 100%, stroke: 1.5pt)
    #v(2cm)
    #text(size: 16pt, weight: "medium", author)
    #v(1fr)
    #line(length: 40%, stroke: 1pt)
    #text(size: 12pt, weight: "semibold")[#smallcaps(if date == none { datetime.today().display("[month repr:long] [year]") } else { date })]
    #v(1cm)
  ])
  pagebreak()

  // ------------------------------------------------------------------ outline
  show outline.entry.where(level: 1): set block(above: 1.5em, below: 0.5em)
  show outline.entry.where(level: 2): set block(above: 1.0em, below: 0.5em)
  outline(title: [Contents #v(10pt)], depth: 2, indent: 0.5cm)
  pagebreak()
  counter(page).update(1)

  // ------------------------------------------------------------------ body
  set page(
    margin: (left: 1.8cm, right: 1.8cm, top: 2.4cm, bottom: 2cm),
    numbering: "1",
    number-align: bottom + center,
    header: context {
      let page-num = counter(page).get().first()
      let chapter-here = query(heading.where(level: 1)).any(h => h.numbering != none and h.location().page() == here().page())
      if page-num > 1 and not chapter-here {
        let headings = query(selector(heading).before(here()))
        let current = if headings.len() > 0 {
          let h = headings.last()
          if h.numbering != none { [#numbering(h.numbering, ..counter(heading).at(h.location())) #h.body] } else { h.body }
        } else { [] }
        grid(columns: (1fr, auto), text(size: 10pt, weight: 500, current), text(size: 10pt, weight: 500)[#counter(page).display()])
        v(-4pt)
        line(length: 100%, stroke: 0.9pt)
      }
    },
    footer: context {
      if counter(page).get().first() == 1 { align(center, text(size: 10pt, weight: 500)[#counter(page).display()]) }
    },
  )
  body
}
