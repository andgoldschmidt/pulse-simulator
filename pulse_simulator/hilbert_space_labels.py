import functools
import numpy as np


def char_kron(a, b):
    """
    Arguments:
        a (List[Char]) -- List of characters.
        b (List[Char]) -- List of characters.

    Returns:
        Kronecker product of character lists.
    """
    a_arr = np.char.array(a)
    b_arr = np.char.array(b)
    return (a_arr[:, None] + b_arr).flatten()


def hilbert_space_basis(levels):
    """Construct labels for the Hilbert space basis specified by the list
     of energy levels.

    Arguments:
        levels (List[Int]) -- List of integers specifying energy levels.

    Returns:
        (List[Str]) List of labels.
    """
    str_levels = [[str(j) for j in range(n)] for n in levels]
    return functools.reduce(char_kron, str_levels)


def print_wavefunction(y, basis, tol=1e-9):
    """Print a wavefunction.

    Arguments:
        y (NumPy.ndarray) -- Vector of coefficients
        basis (List[Str]) -- Vector of labels
    """
    for val, ket in zip(y.data, basis):
        if abs(val) > tol:
            print(f"{val:+.2f} |{ket}>")


def print_density_matrix(y, basis, tol=1e-9):
    """Print a density matrix.

    Arguments:
        y (NumPy.ndarray) -- Matrix of coefficients
        basis (List[Str]) -- Vector of labels
    """
    dm_basis = '|' + basis[:, None] + '><' + basis + '|'
    for row_val, row_ket in zip(y.data, dm_basis):
        row_str = ""
        for rowcol_val, rowcol_ket in zip(row_val, row_ket):
            if abs(rowcol_val) > tol:
                row_str += f"{rowcol_val:+.2f} {rowcol_ket}  "
        if row_str:
            print(row_str)
