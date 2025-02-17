window.MathJax = {
    tex: {
      inlineMath: [["\\(", "\\)"]],
      displayMath: [["\\[", "\\]"]],
      processEscapes: true,
      processEnvironments: true,
      macros: {
        ir: "{I}^{R}",
        il: "{I}^{L}",
        oR: "C_{O}^{R}",
        oL: "C_{O}^{L}",
        br: "{\\bf{r}}",
        bk: "{\\bf{k}}",
        bz: "{\\bf{z}}",
        bR: "{\\bf{R}}",
        ba: "{\\bf{a}}",
        bG: "{\\bf{G}}",
        bT: "{\\bf{T}}",
        bb: "{\\bf{b}}",
        bq: "{\\bf{q}}",
        kbz: "{{k}_{\\rm BZ}}",
        Hhat: "\\hat{\\mathbf{H}}",
        That: "\\hat{\\mathbf{T}}",
        kp: "\\kappa",
        bkp: "\\boldsymbol{\\kappa}"
      }
    },
    options: {
      ignoreHtmlClass: ".*|",
      processHtmlClass: "arithmatex"
    }
  };
  
  document$.subscribe(() => { 
    MathJax.startup.output.clearCache()
    MathJax.typesetClear()
    MathJax.texReset()
    MathJax.typesetPromise()
  })