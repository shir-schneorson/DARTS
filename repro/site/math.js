// Renders the page's TeX with KaTeX, using the paper's macros.
renderMathInElement(document.body, {
  delimiters: [
    { left: "\\[", right: "\\]", display: true },
    { left: "\\(", right: "\\)", display: false },
  ],
  macros: {
    "\\bt": "\\mathbf{t}", "\\bx": "\\mathbf{x}", "\\br": "\\mathbf{r}", "\\by": "\\mathbf{y}",
    "\\bc": "\\mathbf{c}", "\\bs": "\\mathbf{s}", "\\be": "\\mathbf{e}", "\\bz": "\\mathbf{z}",
    "\\bb": "\\mathbf{b}", "\\bJ": "\\mathbf{J}", "\\bI": "\\mathbf{I}", "\\bC": "\\mathbf{C}",
    "\\bmu": "\\boldsymbol{\\mu}", "\\beps": "\\boldsymbol{\\epsilon}", "\\shat": "\\hat{\\mathbf{s}}",
    "\\E": "\\mathbb{E}", "\\R": "\\mathbb{R}", "\\T": "\\mathsf{T}", "\\PD": "P_{\\mathrm{D}}",
    "\\PF": "P_{\\mathrm{FA}}", "\\Normal": "\\mathcal{N}", "\\Var": "\\operatorname{Var}",
    "\\Reg": "\\mathcal{R}", "\\sd": "\\operatorname{sd}",
  },
  throwOnError: false,
});
