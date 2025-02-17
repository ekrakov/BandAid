# Introduction to bands 

# Free particle solution

## Bloch's theorem 
Our lattice is periodic under translation by a lattice vector $\bR$. While it is not immediately clear how this will play into our final solution, it is helpful to start by introducing a translation operator $\That$ such that 

$$
    \That \psi(\br)=\psi(\br+\bR)
$$

The eigenspectrum of this operator seems a little easier to understand than that of $\Hhat$. Because our lattice is periodic in $\bR$, implying $\Hhat(\br + \bR) = \Hhat(\br)$, we can show that the operators $\That$ and $\Hhat$ commute:


$$
    \That\Hhat(\br)\psi(\br) = \Hhat(\br + \bR)\psi(\br + \bR) = \Hhat(\br)\That\psi(\br)
$$


By a well-known property, showing that $\That$ and $\Hhat$ commute under periodicity $\bR$ implies that they share a common set of eigenfunctions. Finding the eigenfunctions of $\That$ then equates to solving for the eigenfunctions of $\Hhat$, a powerful simplification.

We now wish to determine the form of the eigenfunctions of $\That$. For example, eigenfunctions of eigenvalue $\lambda=1$ are totally periodic in $\bR$, such that $f(\br)$  does not change when translated by $\bR$:

$$
\That f(\br)=f(\br+\bR)=f(\br)
$$


More generally, consider a function that is not trivially periodic but  satisfies $\psi(\br+\bR)=\lambda \psi(\br)$. This behavior is well described by an exponential, but because we are describing a quantum system, it has to be an exponential that is normalizable, i.e., a complex exponential of the form $\psi = e^{i\bk\cdot\br}$. We apply $\That$ to find its eigenvalue: 

\[
   \That e^{i\bk\cdot\br} = e^{i(\bk \cdot \br + \bk \cdot \bR)} = 
\underbrace{e^{i \bk \cdot \bR}}_{\lambda} e^{i \bk \cdot \br}
\]

Our final eigenstate $\psi$ can be either periodic or of the form of the complex exponential. In general, any eigenfunction of $\That$ can be written

$$
    \psi(\br)=e^{i\bk \cdot \br}f(\br)
$$

%
where once again, $f(\br) = f(\br + \bR)$, and $e^{i\bk \cdot \br}$ is a consequence of the translational invariance of the system. As before, we can find the eigenvalue:

$$
    \That \psi(\br) = e^{i \bk \cdot \bR} e^{i \bk \cdot \br} f(\br+\bR)=e^{i \bk \cdot \bR} e^{i \bk \cdot \br} f(\br)
$$

Now we reiterate that the eigenfunction $\psi$ also belongs to $\Hhat$---we've exploited the commutativity of the two operators to find a solution for our periodic system simply by considering $\That$. This is Bloch's theorem. These so-called \textit{Bloch waves} can be written in full as

$$
\psi_{n\bk}(\br)=e^{i \bk \cdot \br} f_{n\bk}(\br)
$$


<!-- https://ekrakov-bandaid.streamlit.app/?embed_options=show_toolbar,light_theme,show_colored_line -->




<!-- <iframe src="https://ekrakov-bandaid.streamlit.app/?embed=true" width="100%" height="600px" style="border:none;"></iframe> -->

<iframe src="https://ekrakov-bandaid.streamlit.app/?embed=true&embed_options=show_toolbar"
        style="width: 90%; height: 85vh; border: none;"></iframe>

<!-- <!-- 
<div style="width: 100%; display: flex; justify-content: center; align-items: center;">
  <iframe id="streamlit-iframe" 
          src="https://ekrakov-bandaid.streamlit.app/?embed=true&embed_options=show_toolbar"
          style="width: 90%; height: 95vh; border: none;"></iframe>
</div>

<script>
  window.onload = function() {
    var iframe = document.getElementById("streamlit-iframe");
    iframe.style.height = window.innerHeight * 0.95 + "px";  // 95% of viewport height
  };
</script> -->


<!-- 


<div style="width: 100%; display: flex; justify-content: center; align-items: center;">
  <iframe id="streamlit-iframe" 
          src="https://ekrakov-bandaid.streamlit.app/?embed=true&embed_options=show_toolbar"
          style="width: 90%; height: 95vh; border: none;"></iframe>
</div>

<script>
  window.onload = function() {
    var iframe = document.getElementById("streamlit-iframe");
    setTimeout(function() {
      iframe.src = iframe.src;  // Reload iframe to trigger proper toolbar loading
    }, 1000);
  };
</script> --> -->
