We seek the wavefunction $\psi$ to the schrodinger equation however we do not know what it looks like. We do know that we can represent it as a sum over some othonormal basis functions denoted as $\phi_i$

$$ \psi = \sum _i^{n}c_i\phi_i $$

The full wavefunction is optained once we have our coefficients $c_i$

Our goal is to solve the s.e. $H\psi=E\psi$. We can express this in terms of our basis functions 

$$ \hat{H}\sum_i^{\infty}c_i\phi_j=E\sum_i^{\infty}c_i\phi_i$$

given that we chose orthonormal basis functions, we can multiply by $\phi_i*$ and integrate 


$$ \hat{H}\sum_i^{\infty}c_i\phi_j=E\sum_i^{\infty}c_i\phi_i$$


resulting in 

$$\sum_i^{\infty}H_{mn}c_n=Ec_m$$




We can convert this to a matrix equation 

$$H_{ij}=\int\phi_i\hat{H}\phi_j dx$$

Our hamiltonian matrix is thus 

$$
\tilde{H} =
\begin{bmatrix}
H_{11} & H_{12} & \cdots & H_{1N} \\
H_{21} & H_{22} & \cdots & H_{2N} \\
\vdots & \vdots & \ddots & \vdots \\
H_{N1} & H_{N2} & \cdots & H_{NN}
\end{bmatrix}
$$

then our wavefunction $\psi$ 

<!-- <!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quantum Well Wavefunctions</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
</head>
<body>
    <h1>Quantum Well Wavefunctions</h1>

    <label for="bHeight">Barrier Height:</label>
    <input type="range" id="bHeight" min="10" max="50" step="1" value="30" oninput="updateLabel('bHeightLabel', this.value); updatePlot();">
    <span id="bHeightLabel">30</span>

    <label for="bWidth">Barrier Width:</label>
    <input type="range" id="bWidth" min="0.1" max="2" step="0.1" value="0.5" oninput="updateLabel('bWidthLabel', this.value); updatePlot();">
    <span id="bWidthLabel">0.5</span>

    <svg id="plot" width="800" height="500"></svg>

    <script>
        function updateLabel(labelId, value) {
            document.getElementById(labelId).textContent = value;
        }

        async function updatePlot() {
            let bHeight = parseFloat(document.getElementById("bHeight").value).toFixed(1);
            let bWidth = parseFloat(document.getElementById("bWidth").value).toFixed(1);
            let key = `${bHeight}_${bWidth}`;

            let jsonPath = "/wavefunctions.json";
            console.log("Fetching JSON from:", jsonPath, "with key:", key);

            try {
                let response = await fetch(jsonPath);
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }

                let data_store = await response.json();

                if (!(key in data_store)) {
                    console.error("No precomputed data for this parameter set:", key);
                    return;
                }

                let data = data_store[key];
                let x_vals = data.x_vals;
                let potential = data.potential;
                let wavefunctions = data.wavefunctions;
                let eigenvalues = data.eigenvalues;

                // Ensure arrays are not empty
                if (!x_vals.length || !wavefunctions.length || !eigenvalues.length) {
                    console.error("Data arrays are empty. Cannot plot.");
                    return;
                }

                let svg = d3.select("#plot"),
                    width = +svg.attr("width"),
                    height = +svg.attr("height");

                let margin = { top: 20, right: 20, bottom: 50, left: 50 };
                let plotWidth = width - margin.left - margin.right;
                let plotHeight = height - margin.top - margin.bottom;

                let xScale = d3.scaleLinear()
                    .domain([Math.min(...x_vals), Math.max(...x_vals)])
                    .range([margin.left, plotWidth]);

                let yScale = d3.scaleLinear()
                    .domain([
                        Math.min(...potential, ...eigenvalues) - 5,
                        Math.max(...potential, ...eigenvalues) + 5
                    ])
                    .range([plotHeight, margin.top]);

                // Clear old plots
                svg.selectAll("*").remove();

                // Plot potential function
                svg.append("path")
                    .datum(x_vals.map((x, i) => ({ x: x, y: potential[i] })))
                    .attr("fill", "none")
                    .attr("stroke", "green")
                    .attr("stroke-width", 2)
                    .attr("d", d3.line().x(d => xScale(d.x)).y(d => yScale(d.y)));

                // Plot wavefunctions at their energy levels
                wavefunctions.forEach((wave, i) => {
                    svg.append("path")
                        .datum(x_vals.map((x, j) => ({ x: x, y: wave[j] * 5 + eigenvalues[i] })))
                        .attr("fill", "none")
                        .attr("stroke", d3.schemeCategory10[i % 10])
                        .attr("stroke-width", 1.5)
                        .attr("d", d3.line().x(d => xScale(d.x)).y(d => yScale(d.y)));
                });

                // Add axes
                svg.append("g")
                    .attr("transform", `translate(0, ${plotHeight})`)
                    .call(d3.axisBottom(xScale));

                svg.append("g")
                    .attr("transform", `translate(${margin.left}, 0)`)
                    .call(d3.axisLeft(yScale));

            } catch (error) {
                console.error("Error loading JSON:", error);
            }
        }

        updatePlot(); // Run automatically on page load
    </script>
</body>
</html>
 -->
<!-- 
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quantum Well Wavefunctions</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
</head>
<body>
    <h1>Quantum Well Wavefunctions</h1>

    <label for="nWells">Number of Wells:</label>
    <input type="range" id="nWells" min="1" max="3" step="1" value="1" oninput="updateLabel('nWellsLabel', this.value); updatePlot();">
    <span id="nWellsLabel">1</span>

    <label for="bHeight">Well Depth:</label>
    <input type="range" id="bHeight" min="5" max="15" step="5" value="5" oninput="updateLabel('bHeightLabel', this.value); updatePlot();">
    <span id="bHeightLabel">5</span>

    <label for="bWidth">Barrier Width:</label>
    <input type="range" id="bWidth" min="0.1" max="2" step="0.1" value="0.5" oninput="updateLabel('bWidthLabel', this.value); updatePlot();">
    <span id="bWidthLabel">0.5</span>

    <!-- <svg id="plot" width="100" height="300"></svg> -->
    <svg id="plot" viewBox="0 0 800 500" preserveAspectRatio="xMidYMid meet" style="width: 100%; height: 500px;"></svg>
    <div id="plot-container" style="width: 100%; max-width: 900px; height: auto; margin: auto; position: relative;">
    <svg id="plot" viewBox="0 0 800 500" preserveAspectRatio="xMidYMid meet" style="width: 100%; height: auto;"></svg>
</div>

    <script>
        function updateLabel(labelId, value) {
            document.getElementById(labelId).textContent = value;
        }

        async function updatePlot() {
            let nWells = document.getElementById("nWells").value;
            let bHeight = document.getElementById("bHeight").value;
            let bWidth = parseFloat(document.getElementById("bWidth").value).toFixed(1);
            let key = `${nWells}_wells_${bHeight}_depth`;

            let jsonPath = window.location.origin + "/wavefunctions.json"; // Ensure correct file path
            console.log("Fetching JSON from:", jsonPath, "with key:", key);

            try {
                let response = await fetch(jsonPath);
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }

                let data_store = await response.json();

                if (!(key in data_store)) {
                    console.error("No precomputed data for this parameter set:", key);
                    return;
                }

                let data = data_store[key];
                let x_vals = data.x_vals;
                let potential = data.potential;
                let wavefunctions = data.wavefunctions;
                let eigenvalues = data.eigenvalues;

                if (!x_vals.length || !wavefunctions.length || !eigenvalues.length) {
                    console.error("Data arrays are empty. Cannot plot.");
                    return;
                }

                let svg = d3.select("#plot"),
                    width = +svg.attr("width"),
                    height = +svg.attr("height");

                let margin = { top: 20, right: 20, bottom: 50, left: 50 };
                let plotWidth = width - margin.left - margin.right;
                let plotHeight = height - margin.top - margin.bottom;

                let xScale = d3.scaleLinear()
                    .domain([Math.min(...x_vals), Math.max(...x_vals)])
                    .range([margin.left, plotWidth]);

                let yScale = d3.scaleLinear()
                    .domain([
                        Math.min(...potential, ...eigenvalues) - 5,
                        Math.max(...potential, ...eigenvalues) + 5
                    ])
                    .range([plotHeight, margin.top]);

                svg.selectAll("*").remove();

                svg.append("path")
                    .datum(x_vals.map((x, i) => ({ x: x, y: potential[i] })))
                    .attr("fill", "none")
                    .attr("stroke", "green")
                    .attr("stroke-width", 2)
                    .attr("d", d3.line().x(d => xScale(d.x)).y(d => yScale(d.y)));

                wavefunctions.forEach((wave, i) => {
                    svg.append("path")
                        .datum(x_vals.map((x, j) => ({ x: x, y: wave[j] * 5 + eigenvalues[i] })))
                        .attr("fill", "none")
                        .attr("stroke", d3.schemeCategory10[i % 10])
                        .attr("stroke-width", 1.5)
                        .attr("d", d3.line().x(d => xScale(d.x)).y(d => yScale(d.y)));
                });

                svg.append("g")
                    .attr("transform", `translate(0, ${plotHeight})`)
                    .call(d3.axisBottom(xScale));

                svg.append("g")
                    .attr("transform", `translate(${margin.left}, 0)`)
                    .call(d3.axisLeft(yScale));

            } catch (error) {
                console.error("Error loading JSON:", error);
            }
        }

        updatePlot();
    </script>
</body>
</html> -->
