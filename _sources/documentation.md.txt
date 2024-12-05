# Documentation Instructions 


To add a page, create a markdown file in the docs directory.



```bash
docs/<page_title_here>.md
```

After you create the page, open ```docs/index.rst```

 add ```<page_title_here>.md``` to ```docs/index.rst```


```bash
.. docs/index.rst   
.. include:: ../README.rst

Welcome to Open Dis documentation!
===================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   <page_title_here>.md
   examples.md
   installation.md
```

