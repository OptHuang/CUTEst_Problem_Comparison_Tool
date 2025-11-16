# CUTEst Problem Comparison Tool

## Purpose
This repository aims to provide a simple tool for comparing instances of CUTEst problems in PyCUTEst and S2MPJ (both Python and MATLAB implementations). It transforms CUTEst problems into `Problem` instances defined in OptiProfiler (www.optprof.com) and compares their outputs. Specifically, it checks the following aspects:

- **Basic properties**: Problem name, dimension (`n`), and problem type (`ptype`)
- **Constraint counts**: Number of bound constraints (`mb`), linear inequality constraints (`m_linear_ub`), linear equality constraints (`m_linear_eq`), nonlinear inequality constraints (`m_nonlinear_ub`), and nonlinear equality constraints (`m_nonlinear_eq`)
- **Variable bounds**: Lower bounds (`xl`) and upper bounds (`xu`)
- **Initial point**: Starting point (`x0`) for optimization
- **Linear constraints**: Coefficient matrices and right-hand side vectors for linear inequalities (`aub`, `bub`) and equalities (`aeq`, `beq`)
- **Function evaluations**: Objective function values (`fun`) and nonlinear constraint values (`cub`, `ceq`) at multiple randomly sampled points

The comparison ensures that both implementations produce consistent problem formulations, which is crucial for reliable benchmarking of optimization solvers.

## Workflow

**Step 1:** Clone this repository, clone and install OptiProfiler under this repository (both MATLAB and Python versions). PyCUTEst is assumed to be already installed.

**Step 2:** (Optional) If you have modified S2MPJ problem files (e.g., `HS67.m` or `HS67.py`), place them in `./optiprofiler/problems/s2mpj/src/matlab_problems/` or `./optiprofiler/problems/s2mpj/src/python_problems/` respectively.

**Step 3:** In MATLAB, run `get_s2mpj_matlab_pbinfo.m` from the repository root directory. Remember to modify `pb_names = {'HS67', 'HS68', 'HS69', 'HS85', 'HS88', 'HS89', 'HS90', 'HS91', 'HS92'};` to include the problems you want to check. This generates `.h5` files in the current directory.

**Step 4:** Run `compare_pycutest_s2mpj_python.py` to compare PyCUTEst and S2MPJ Python versions. Remember to modify `pb_names` in the script. The comparison results will be displayed in the terminal.

**Step 5:** Run `compare_s2mpj_matlab_python.py` to compare S2MPJ MATLAB and Python versions. Remember to modify `pb_names` in the script. The comparison results will be displayed in the terminal.
