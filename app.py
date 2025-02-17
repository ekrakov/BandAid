from fastapi import FastAPI
import numpy as np
import scipy.linalg
from numba import jit


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows requests from any frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@jit(nopython=True)
def fast_diagonalization(H):
    return np.linalg.eigh(H)

@app.get("/compute")
def compute_eigenstates(bHeight: float = 30, bWidth: float = 0.5):
    nMax = 20  # 20×20 matrix
    H_dynamic = np.zeros((nMax, nMax))
    eigenvalues, _ = fast_diagonalization(H_dynamic)
    return {"eigenvalues": eigenvalues.tolist()}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=True)

# Speed up diagonalization with Numba
@jit(nopython=True)
def fast_diagonalization(H):
    return np.linalg.eigh(H)

@app.get("/compute")
def compute_eigenstates(bHeight: float = 30, bWidth: float = 0.5):
    nMax = 20  # Number of basis functions (20x20 matrix)
    b = 2 + bWidth * 2  # System width

    # Define sine basis functions
    def basis(n, x, b):
        return np.sqrt(2 / b) * np.sin(n * np.pi * x / b)

    # Define potential function
    def V_dynamic(x):
        wellStart = bWidth
        wellEnd = wellStart + 1
        return -bHeight if wellStart <= x < wellEnd else 0

    # Construct Hamiltonian matrix
    H_dynamic = np.zeros((nMax, nMax))
    for m in range(1, nMax + 1):
        for n in range(1, nMax + 1):
            if m == n:
                H_dynamic[m - 1, n - 1] = (n ** 2 * np.pi ** 2) / (2 * b ** 2)

    # Diagonalize Hamiltonian
    eigenvalues, eigenvectors = fast_diagonalization(H_dynamic)

    return {
        "eigenvalues": eigenvalues.tolist()
    }

# Run with: uvicorn app:app --reload
