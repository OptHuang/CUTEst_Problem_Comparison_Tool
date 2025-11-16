# compare_s2mpj_matlab_python.py

import h5py
import numpy as np
from pathlib import Path
import optiprofiler
from problems.s2mpj.s2mpj_tools import s2mpj_load


def load_matlab_h5(filepath):
    """Load MATLAB-generated HDF5 file."""
    data = {}
    with h5py.File(filepath, 'r') as f:
        # Scalars - use .item() to extract scalar values
        data['name'] = f['name'][()].decode() if isinstance(f['name'][()], bytes) else str(f['name'][()])
        data['n'] = int(f['n'][()].item())
        data['mb'] = int(f['mb'][()].item())
        data['m_linear_ub'] = int(f['m_linear_ub'][()].item())
        data['m_linear_eq'] = int(f['m_linear_eq'][()].item())
        data['m_nonlinear_ub'] = int(f['m_nonlinear_ub'][()].item())
        data['m_nonlinear_eq'] = int(f['m_nonlinear_eq'][()].item())
        data['mlcon'] = int(f['mlcon'][()].item())
        data['mnlcon'] = int(f['mnlcon'][()].item())
        data['mcon'] = int(f['mcon'][()].item())
        
        # Problem type
        ptype_raw = f['ptype'][()]
        if isinstance(ptype_raw, np.ndarray):
            ptype_raw = ptype_raw.item()
        if isinstance(ptype_raw, bytes):
            data['ptype'] = ptype_raw.decode()  # b'n' → 'n'
        else:
            data['ptype'] = str(ptype_raw)
        
        # Vectors/Matrices
        data['x0'] = f['x0'][()].flatten()
        data['xl'] = f['xl'][()].flatten()
        data['xu'] = f['xu'][()].flatten()
        
        # Optional linear constraints
        if 'aub' in f:
            aub_raw = f['aub'][()]
            # Ensure aub is 2D with shape (m, n)
            if aub_raw.ndim == 2 and aub_raw.shape[1] == 1:
                data['aub'] = aub_raw.T  # (n, 1) → (1, n)
            else:
                data['aub'] = aub_raw
        else:
            data['aub'] = None
        data['bub'] = f['bub'][()].flatten() if 'bub' in f else None
        if 'aeq' in f:
            aeq_raw = f['aeq'][()]
            # Ensure aeq is 2D with shape (m, n)
            if aeq_raw.ndim == 2 and aeq_raw.shape[1] == 1:
                data['aeq'] = aeq_raw.T  # (n, 1) → (1, n)
            else:
                data['aeq'] = aeq_raw
        else:
            data['aeq'] = None
        data['beq'] = f['beq'][()].flatten() if 'beq' in f else None
        
        # Sample points and objective function values
        data['X_samples'] = f['X_samples'][()]
        data['fun_values'] = f['fun_values'][()].flatten()
        
        # Optional nonlinear constraint values
        data['cub_values'] = f['cub_values'][()] if 'cub_values' in f else None
        data['ceq_values'] = f['ceq_values'][()] if 'ceq_values' in f else None
    
    return data


def compare_problems(problem_name, h5_filepath, rtol=1e-5, atol=1e-8):
    """
    Compare MATLAB and Python problem implementations.
    
    Parameters
    ----------
    problem_name : str
        Problem name (e.g., 'HS67')
    h5_filepath : str or Path
        Path to MATLAB-generated HDF5 file
    rtol : float
        Relative tolerance for numerical comparison
    atol : float
        Absolute tolerance for numerical comparison
    
    Returns
    -------
    dict
        Comparison results
    """
    print(f"\n{'='*60}")
    print(f"Testing: {problem_name}")
    print(f"{'='*60}")
    
    # Load MATLAB data
    matlab_data = load_matlab_h5(h5_filepath)
    
    # Load Python problem
    python_prob = s2mpj_load(problem_name)
    
    results = {
        'problem_name': problem_name,
        'passed': True,
        'failures': []
    }
    
    # 1. Compare metadata
    print("\n[1] Metadata Comparison")
    print("-" * 40)
    
    metadata_fields = ['n', 'mb', 'm_linear_ub', 'm_linear_eq', 
                       'm_nonlinear_ub', 'm_nonlinear_eq', 
                       'mlcon', 'mnlcon', 'mcon']
    
    for field in metadata_fields:
        matlab_val = matlab_data[field]
        python_val = getattr(python_prob, field)
        
        match = matlab_val == python_val
        symbol = "✓" if match else "✗"
        print(f"{symbol} {field:20s}: MATLAB={matlab_val:6}, Python={python_val:6}")
        
        if not match:
            results['passed'] = False
            results['failures'].append(f"Metadata mismatch: {field}")
    
    # Compare ptype (more lenient)
    matlab_ptype = matlab_data['ptype'].strip()
    python_ptype = python_prob.ptype.strip()
    ptype_match = matlab_ptype == python_ptype
    symbol = "✓" if ptype_match else "✗"
    print(f"{symbol} {'ptype':20s}: MATLAB={matlab_ptype}, Python={python_ptype}")
    if not ptype_match:
        results['passed'] = False
        results['failures'].append(f"Metadata mismatch: ptype")
    
    # 2. Compare initial point and bounds
    print("\n[2] Initial Point & Bounds Comparison")
    print("-" * 40)
    
    vector_fields = [('x0', 'Initial point'),
                     ('xl', 'Lower bounds'),
                     ('xu', 'Upper bounds')]
    
    for field, desc in vector_fields:
        matlab_vec = matlab_data[field]
        python_vec = getattr(python_prob, field)
        
        # Handle inf values in bounds
        match = np.allclose(matlab_vec, python_vec, rtol=rtol, atol=atol, equal_nan=True)
        
        # Calculate max_diff only for finite values
        finite_mask = np.isfinite(matlab_vec) & np.isfinite(python_vec)
        if np.any(finite_mask):
            max_diff = np.max(np.abs(matlab_vec[finite_mask] - python_vec[finite_mask]))
        else:
            max_diff = 0.0
            
        symbol = "✓" if match else "✗"
        print(f"{symbol} {desc:20s}: max_diff={max_diff:.2e}")
        
        if not match:
            results['passed'] = False
            results['failures'].append(f"{desc} mismatch")
    
    # 3. Compare linear constraints
    print("\n[3] Linear Constraints Comparison")
    print("-" * 40)
    
    if matlab_data['aub'] is not None:
        if python_prob.aub is not None:
            aub_match = np.allclose(matlab_data['aub'], python_prob.aub, rtol=rtol, atol=atol)
            bub_match = np.allclose(matlab_data['bub'], python_prob.bub, rtol=rtol, atol=atol)
            
            aub_diff = np.max(np.abs(matlab_data['aub'] - python_prob.aub))
            bub_diff = np.max(np.abs(matlab_data['bub'] - python_prob.bub))
            
            print(f"{'✓' if aub_match else '✗'} A_ub: max_diff={aub_diff:.2e}")
            print(f"{'✓' if bub_match else '✗'} b_ub: max_diff={bub_diff:.2e}")
            
            if not (aub_match and bub_match):
                results['passed'] = False
                results['failures'].append("Linear inequality constraints mismatch")
                # Debug info
                if not aub_match:
                    print(f"    MATLAB A_ub shape: {matlab_data['aub'].shape}")
                    print(f"    Python A_ub shape: {python_prob.aub.shape}")
                    print(f"    MATLAB A_ub:\n{matlab_data['aub']}")
                    print(f"    Python A_ub:\n{python_prob.aub}")
        else:
            print("✗ MATLAB has A_ub/b_ub, Python doesn't")
            results['passed'] = False
            results['failures'].append("Missing linear inequality constraints in Python")
    else:
        print("✓ No linear inequality constraints")
    
    if matlab_data['aeq'] is not None:
        if python_prob.aeq is not None:
            aeq_match = np.allclose(matlab_data['aeq'], python_prob.aeq, rtol=rtol, atol=atol)
            beq_match = np.allclose(matlab_data['beq'], python_prob.beq, rtol=rtol, atol=atol)
            
            aeq_diff = np.max(np.abs(matlab_data['aeq'] - python_prob.aeq))
            beq_diff = np.max(np.abs(matlab_data['beq'] - python_prob.beq))
            
            print(f"{'✓' if aeq_match else '✗'} A_eq: max_diff={aeq_diff:.2e}")
            print(f"{'✓' if beq_match else '✗'} b_eq: max_diff={beq_diff:.2e}")
            
            if not (aeq_match and beq_match):
                results['passed'] = False
                results['failures'].append("Linear equality constraints mismatch")
        else:
            print("✗ MATLAB has A_eq/b_eq, Python doesn't")
            results['passed'] = False
            results['failures'].append("Missing linear equality constraints in Python")
    else:
        print("✓ No linear equality constraints")
    
    # 4. Compare function values at sample points
    print("\n[4] Objective Function Comparison")
    print("-" * 40)
    
    X_samples = matlab_data['X_samples']  # Now shape is (n_samples, n)
    matlab_fun_values = matlab_data['fun_values']
    
    print(f"X_samples shape: {X_samples.shape}")
    print(f"Expected shape: (n_samples, {python_prob.n})")
    
    # Call Python function with each sample point
    python_fun_values = np.array([python_prob.fun(x) for x in X_samples])
    
    fun_match = np.allclose(matlab_fun_values, python_fun_values, rtol=rtol, atol=atol)
    fun_max_diff = np.max(np.abs(matlab_fun_values - python_fun_values))
    fun_mean_diff = np.mean(np.abs(matlab_fun_values - python_fun_values))
    
    print(f"{'✓' if fun_match else '✗'} Objective function:")
    print(f"    max_diff  = {fun_max_diff:.2e}")
    print(f"    mean_diff = {fun_mean_diff:.2e}")
    print(f"    Tested on {len(matlab_fun_values)} sample points")
    
    if not fun_match:
        results['passed'] = False
        results['failures'].append("Objective function values mismatch")
    
    # 5. Compare nonlinear constraint values
    print("\n[5] Nonlinear Constraints Comparison")
    print("-" * 40)
    
    if matlab_data['cub_values'] is not None:
        if python_prob.cub is not None and python_prob.m_nonlinear_ub > 0:
            python_cub_values = np.array([python_prob.cub(x) for x in X_samples])
            
            cub_match = np.allclose(matlab_data['cub_values'], python_cub_values, rtol=rtol, atol=atol)
            cub_max_diff = np.max(np.abs(matlab_data['cub_values'] - python_cub_values))
            cub_mean_diff = np.mean(np.abs(matlab_data['cub_values'] - python_cub_values))
            
            print(f"{'✓' if cub_match else '✗'} Nonlinear inequalities (c_ub):")
            print(f"    max_diff  = {cub_max_diff:.2e}")
            print(f"    mean_diff = {cub_mean_diff:.2e}")
            
            if not cub_match:
                results['passed'] = False
                results['failures'].append("Nonlinear inequality constraints mismatch")
        else:
            print("✗ MATLAB has c_ub, Python doesn't")
            results['passed'] = False
            results['failures'].append("Missing nonlinear inequality constraints in Python")
    else:
        print("✓ No nonlinear inequality constraints")
    
    if matlab_data['ceq_values'] is not None:
        if python_prob.ceq is not None and python_prob.m_nonlinear_eq > 0:
            python_ceq_values = np.array([python_prob.ceq(x) for x in X_samples])
            
            ceq_match = np.allclose(matlab_data['ceq_values'], python_ceq_values, rtol=rtol, atol=atol)
            ceq_max_diff = np.max(np.abs(matlab_data['ceq_values'] - python_ceq_values))
            ceq_mean_diff = np.mean(np.abs(matlab_data['ceq_values'] - python_ceq_values))
            
            print(f"{'✓' if ceq_match else '✗'} Nonlinear equalities (c_eq):")
            print(f"    max_diff  = {ceq_max_diff:.2e}")
            print(f"    mean_diff = {ceq_mean_diff:.2e}")
            
            if not ceq_match:
                results['passed'] = False
                results['failures'].append("Nonlinear equality constraints mismatch")
        else:
            print("✗ MATLAB has c_eq, Python doesn't")
            results['passed'] = False
            results['failures'].append("Missing nonlinear equality constraints in Python")
    else:
        print("✓ No nonlinear equality constraints")
    
    # Summary
    print(f"\n{'='*60}")
    if results['passed']:
        print(f"✓ ALL TESTS PASSED for {problem_name}")
    else:
        print(f"✗ TESTS FAILED for {problem_name}")
        print(f"\nFailures:")
        for failure in results['failures']:
            print(f"  - {failure}")
    print(f"{'='*60}")
    
    return results


def run_all_tests(h5_dir='.', problems=None):
    """
    Run comparison tests for all problems.
    
    Parameters
    ----------
    h5_dir : str or Path
        Directory containing MATLAB HDF5 files
    problems : list of str, optional
        List of problem names to test. If None, test all .h5 files in h5_dir
    
    Returns
    -------
    dict
        Summary of all test results
    """
    h5_dir = Path(h5_dir)
    
    if problems is None:
        # Auto-detect all .h5 files
        h5_files = list(h5_dir.glob('*.h5'))
        problems = [f.stem for f in h5_files]
    
    all_results = []
    
    for problem_name in problems:
        h5_file = h5_dir / f"{problem_name}.h5"
        
        if not h5_file.exists():
            print(f"\n✗ Skipping {problem_name}: {h5_file} not found")
            continue
        
        try:
            result = compare_problems(problem_name, h5_file)
            all_results.append(result)
        except Exception as e:
            print(f"\n✗ Error testing {problem_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                'problem_name': problem_name,
                'passed': False,
                'failures': [f"Exception: {str(e)}"]
            })
    
    # Overall summary
    print("\n" + "="*60)
    print("OVERALL SUMMARY")
    print("="*60)
    
    passed = sum(1 for r in all_results if r['passed'])
    total = len(all_results)
    
    print(f"\nPassed: {passed}/{total} ({100*passed/total:.1f}%)")
    
    if passed < total:
        print(f"\nFailed problems:")
        for r in all_results:
            if not r['passed']:
                print(f"  - {r['problem_name']}: {', '.join(r['failures'])}")
    
    return all_results


if __name__ == '__main__':
    # Test specific problems
    pb_names = ['HS67', 'HS68', 'HS69', 'HS85', 'HS88', 'HS89', 'HS90', 'HS91', 'HS92']
    
    results = run_all_tests(h5_dir='.', problems=pb_names)