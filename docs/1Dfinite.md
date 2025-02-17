# 1D example 

To understand material properties, we need to study how electrons move in solids. While classical mechanics gave us some insight into how electrons move, certain properties, such as heat capcaity, etc can not be propoerly descirbed without quantum mechanics. The Schrödinger equation describes the quantum behavior of an electron in a given potential. However, solving the Schrödinger equation for a solid, where multiple periodic potentials, is extremely challenging. This tutorial introduces the concept of band structure—the energy eigenstates of electrons in a periodic potential.

## Finite square well
Before tackling the Schrödinger equation in a system with multiple potentials, it's important to first understand its solutions for a single, isolated potential. Suppose we make a huge simplification and assume the potential an electron feels due to the nucleus is given by a 1D square well potential. 

We are interested in solving the  1D time independent schrodinger equation

$$
\left[ -\frac{\hbar^2}{2m} \frac{d^2}{dx^2} + V(x) \right] \Psi(x) = E \Psi(x)
$$

subject to a simple finite square well potential 

$$
V(x) =
\begin{cases} 
0, & x < -a \\ 
-V_0, & -a \leq x \leq a\\
0, & x > a  \\ 
\end{cases}
$$

The solutions can be either bound states, ($E<0$) or scattering states $E>0$. Both are important for understanding electronic behavior. For simplicity, we we start by describing the bound state solutions. 

$$
\Psi(x) =
\begin{cases} 
A e^{\kappa x}, & x < -a \\ 
B \cos(k x) + C \sin(k x), & -a \leq x \leq a \\ 
D e^{-\kappa x}, & x > a
\end{cases}
$$

where $\kappa$ and $k$ are


\[
\kappa = \sqrt{\frac{2m |E|}{\hbar^2}}
\]

\[
k = \sqrt{\frac{2m (V_0 + E)}{\hbar^2}}
\]

To find the allowed energy levels, we must enforce  continuity of $\psi(x)$ and $\psi'(x)$ at $x=-a$ and $x=a$

Due to the symmetry of the problem, we realize the solutions inside the well are either even or odd. We enforce the boundary conditions for each one respectively. This results in the following transidental equations 

for even states

$$
\kappa = k\tan(ka)
$$

for odd states 

$$
\kappa = -k\cot(ka)
$$




<!-- ## Project layout

    mkdocs.yml    # The configuration file.
    docs/
        index.md  # The documentation homepage.
        ...       # Other markdown pages, images and other files.
 -->



<!-- 
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <label for="V0">V0:</label>
    <input type="range" id="V0" min="5" max="50" step="1" value="30" oninput="updatePlot()">
    <span id="V0_value">30</span>
    
    <label for="a">a:</label>
    <input type="range" id="a" min="0.5" max="5" step="0.1" value="1" oninput="updatePlot()">
    <span id="a_value">1</span>
    
    <div id="plot"></div>
    
    <script>
        function computeData(V0, a) {
            let E_values = [];
            let kap_values = [];
            let tan_values = [];
            let cot_values = [];
            
            for (let E = -50; E < 0; E += 0.1) {
                let k = Math.sqrt(2 * (V0 + E));
                let kappa = Math.sqrt(2 * Math.abs(E));
                let tanVal = k * Math.tan(k * a);
                let cotVal = -k / Math.tan(k * a);
                
                E_values.push(E);
                kap_values.push(kappa);
                
                if (Math.abs(tanVal) < 10) {
                    tan_values.push(tanVal);
                } else {
                    tan_values.push(null); // Prevents unwanted connections
                }
                
                if (Math.abs(cotVal) < 10) {
                    cot_values.push(cotVal);
                } else {
                    cot_values.push(null); // Prevents unwanted connections
                }
            }

            return { E_values, kap_values, tan_values, cot_values };
        }
        
        function updatePlot() {
            let V0 = parseFloat(document.getElementById("V0").value);
            let a = parseFloat(document.getElementById("a").value);
            document.getElementById("V0_value").innerText = V0;
            document.getElementById("a_value").innerText = a;
            
            let { E_values, kap_values, tan_values, cot_values } = computeData(V0, a);
            
            let traceKappa = {
                x: E_values,
                y: kap_values,
                mode: 'lines',
                name: 'kappa(E)',
                line: { color: 'blue' }
            };
            
            let traceTan = {
                x: E_values,
                y: tan_values,
                mode: 'lines',
                name: 'k tan(ka)',
                marker: { color: 'orange' }
            };
            
            let traceCot = {
                x: E_values,
                y: cot_values,
                mode: 'lines',
                name: '-k cot(ka)',
                marker: { color: 'green' }
            };
            
            let maxKappa = Math.max(...kap_values);

            let layout = {
                title: 'Quantum Well Bound States',
                xaxis: {
                    title: 'Energy E (eV)',
                    showgrid: true,
                    range: [-V0, 0] 
                },
           
                yaxis: {
                title: '',
                showgrid: true,
                range: [0, maxKappa] 
                }

            };
            
            Plotly.newPlot('plot', [traceKappa, traceTan, traceCot], layout);
        }
        
        updatePlot();
    </script>
</body>
</html> -->

<!-- 
let maxKappa = Math.max(...k_values);

let layout = {
    title: 'Quantum Well Bound States',
    xaxis: {
        title: 'Energy E (eV)',
        showgrid: true
    },
    yaxis: {
        title: 'Function Value',
        showgrid: true,
        range: [0, maxKappa] // Set y-axis from 0 to max of kappa(E)
    }
}; -->






<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <label for="V0">V0:</label>
    <input type="range" id="V0" min="1" max="20" step="1" value="30" oninput="updatePlot()">
    <span id="V0_value">30</span>
    
    <label for="a">a/2:</label>
    <input type="range" id="a" min="0.5" max="3" step="0.1" value="1" oninput="updatePlot()">
    <span id="a_value">1</span>
    
    <div id="plot"></div>
    <div>Eigenvalues from Intersection: <span id="energy_levels"></span></div>
    
    <script>
        function computeData(V0, a) {
            let E_values = [];
            let k_values = [];
            let tan_values = [];
            let cot_values = [];
            let intersection_points = [];
            
            for (let E = -50; E < 0; E += 0.0005) {
                let k = Math.sqrt(2 * (V0 + E));
                let kappa = Math.sqrt(2 * Math.abs(E));
                let tanVal = k * Math.tan(k * a);
                let cotVal = -k / Math.tan(k * a);
                
                E_values.push(E);
                k_values.push(kappa);
                
                if (Math.abs(tanVal) < 10) {
                    tan_values.push(tanVal);
                } else {
                    tan_values.push(null); // Prevents unwanted connections
                }
                
                if (Math.abs(cotVal) < 10) {
                    cot_values.push(cotVal);
                } else {
                    cot_values.push(null); // Prevents unwanted connections
                }
                
                // Find intersection points
                if (Math.abs(kappa - tanVal) < 0.0006|| Math.abs(kappa - cotVal) < 0.0006) {
                    intersection_points.push({ E: E, kappa: kappa });
                }
            }
            return { E_values, k_values, tan_values, cot_values, intersection_points };
        }
        
        function updatePlot() {
            let V0 = parseFloat(document.getElementById("V0").value);
            let a = parseFloat(document.getElementById("a").value);
            document.getElementById("V0_value").innerText = V0;
            document.getElementById("a_value").innerText = a;
            
            let { E_values, k_values, tan_values, cot_values, intersection_points } = computeData(V0, a);
            
            let traceKappa = {
                x: E_values,
                y: k_values,
                mode: 'lines',
                name: 'kappa(E)',
                line: { color: 'blue' }
            };
            
            let traceTan = {
                x: E_values,
                y: tan_values,
                mode: 'lines',
                name: 'k tan(ka)',
                line: { color: 'orange', width: 2 }
            };
            
            let traceCot = {
                x: E_values,
                y: cot_values,
                mode: 'lines',
                name: '-k cot(ka)',
                line: { color: 'green', width: 2 }
            };
            
            let intersectionTrace = {
                x: intersection_points.map(p => p.E),
                y: intersection_points.map(p => p.kappa),
                mode: 'markers+text',
                text: intersection_points.map(p => p.E.toFixed(2) + ' eV'),
                textposition: 'top center',
                name: 'bound state energy!',
                marker: { color: 'red', size: 8 }
            };
            

            let maxKappa = Math.max(...k_values);

            let layout = {
                title: 'Finite Square Well Bound States XD',
                xaxis: {
                    title: 'Energy (eV)',
                    showgrid: true,
                    range: [-V0, 0]
                },

                yaxis: {
                title: ' ',
                showgrid: true,
                range: [0, maxKappa] 
                },

                autosize: true,
                responsive: true
            };
            
            let plotDiv = document.getElementById("plot");
            plotDiv.style.width = "75%";  
            plotDiv.style.height = "500px";  
            plotDiv.style.margin = "auto"; 
            Plotly.newPlot('plot', [traceKappa, traceTan, traceCot, intersectionTrace], layout);
            
            // Display intersection energies
            document.getElementById("energy_levels").innerText = intersection_points.map(p => p.E.toFixed(2)).join(", ");
        }
        
        updatePlot();
    </script>
</body>
</html>


## Two square wells

Our final goal is to understand electronic behavior in solids. We take our previous example one step closer by considering the behavior of a single electron in the potential induced by two nuclei spaced apart by a distannce $a$. This is essentially the $H_2$ molecule, but we follow with the simplification that the $H_2$ atom 



