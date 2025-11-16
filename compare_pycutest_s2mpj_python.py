import numpy as np
import optiprofiler
from problems.pycutest.pycutest_tools import pycutest_load
from problems.s2mpj.s2mpj_tools import s2mpj_load

def compare_problems(p_pycutest, p_s2mpj, rtol=1e-9, atol=1e-12):
    """
    Compare two problem instances (pycutest vs s2mpj).
    
    Returns:
        dict: Comparison results with 'passed' (bool) and 'failures' (list)
    """
    results = {
        'passed': True,
        'failures': []
    }
    
    print(f"\n{'='*60}")
    print(f"Comparing: {p_pycutest.name}")
    print(f"{'='*60}\n")
    
    # Basic properties
    print("Basic Properties:")
    print("-" * 40)
    
    name_match = p_pycutest.name == p_s2mpj.name
    print(f"{'✓' if name_match else '✗'} name: pycutest={p_pycutest.name}, s2mpj={p_s2mpj.name}")
    if not name_match:
        results['passed'] = False
        results['failures'].append("Name mismatch")
    
    n_match = p_pycutest.n == p_s2mpj.n
    print(f"{'✓' if n_match else '✗'} n: pycutest={p_pycutest.n}, s2mpj={p_s2mpj.n}")
    if not n_match:
        results['passed'] = False
        results['failures'].append("Dimension mismatch")
    
    ptype_match = p_pycutest.ptype == p_s2mpj.ptype
    print(f"{'✓' if ptype_match else '✗'} ptype: pycutest={p_pycutest.ptype}, s2mpj={p_s2mpj.ptype}")
    if not ptype_match:
        results['passed'] = False
        results['failures'].append("Problem type mismatch")
    
    # Constraint counts
    print("\nConstraint Counts:")
    print("-" * 40)
    
    mb_match = p_pycutest.mb == p_s2mpj.mb
    print(f"{'✓' if mb_match else '✗'} mb: pycutest={p_pycutest.mb}, s2mpj={p_s2mpj.mb}")
    if not mb_match:
        results['passed'] = False
        results['failures'].append("Bound constraints count mismatch")
    
    m_linear_ub_match = p_pycutest.m_linear_ub == p_s2mpj.m_linear_ub
    print(f"{'✓' if m_linear_ub_match else '✗'} m_linear_ub: pycutest={p_pycutest.m_linear_ub}, s2mpj={p_s2mpj.m_linear_ub}")
    if not m_linear_ub_match:
        results['passed'] = False
        results['failures'].append("Linear inequality constraints count mismatch")
    
    m_linear_eq_match = p_pycutest.m_linear_eq == p_s2mpj.m_linear_eq
    print(f"{'✓' if m_linear_eq_match else '✗'} m_linear_eq: pycutest={p_pycutest.m_linear_eq}, s2mpj={p_s2mpj.m_linear_eq}")
    if not m_linear_eq_match:
        results['passed'] = False
        results['failures'].append("Linear equality constraints count mismatch")
    
    m_nonlinear_ub_match = p_pycutest.m_nonlinear_ub == p_s2mpj.m_nonlinear_ub
    print(f"{'✓' if m_nonlinear_ub_match else '✗'} m_nonlinear_ub: pycutest={p_pycutest.m_nonlinear_ub}, s2mpj={p_s2mpj.m_nonlinear_ub}")
    if not m_nonlinear_ub_match:
        results['passed'] = False
        results['failures'].append("Nonlinear inequality constraints count mismatch")
    
    m_nonlinear_eq_match = p_pycutest.m_nonlinear_eq == p_s2mpj.m_nonlinear_eq
    print(f"{'✓' if m_nonlinear_eq_match else '✗'} m_nonlinear_eq: pycutest={p_pycutest.m_nonlinear_eq}, s2mpj={p_s2mpj.m_nonlinear_eq}")
    if not m_nonlinear_eq_match:
        results['passed'] = False
        results['failures'].append("Nonlinear equality constraints count mismatch")
    
    # Vectors
    print("\nVectors:")
    print("-" * 40)
    
    x0_match = np.allclose(p_pycutest.x0, p_s2mpj.x0, rtol=rtol, atol=atol)
    x0_diff = np.max(np.abs(p_pycutest.x0 - p_s2mpj.x0))
    print(f"{'✓' if x0_match else '✗'} x0: max_diff={x0_diff:.2e}")
    if not x0_match:
        results['passed'] = False
        results['failures'].append("Initial point mismatch")
        print(f"    pycutest x0: {p_pycutest.x0}")
        print(f"    s2mpj x0:    {p_s2mpj.x0}")
    
    xl_match = np.allclose(p_pycutest.xl, p_s2mpj.xl, rtol=rtol, atol=atol)
    xl_diff = np.max(np.abs(p_pycutest.xl - p_s2mpj.xl))
    print(f"{'✓' if xl_match else '✗'} xl: max_diff={xl_diff:.2e}")
    if not xl_match:
        results['passed'] = False
        results['failures'].append("Lower bounds mismatch")
    
    xu_match = np.allclose(p_pycutest.xu, p_s2mpj.xu, rtol=rtol, atol=atol)
    xu_diff = np.max(np.abs(p_pycutest.xu - p_s2mpj.xu))
    print(f"{'✓' if xu_match else '✗'} xu: max_diff={xu_diff:.2e}")
    if not xu_match:
        results['passed'] = False
        results['failures'].append("Upper bounds mismatch")
    
    # Linear constraints
    print("\nLinear Constraints:")
    print("-" * 40)

    def safe_max_diff(arr1, arr2):
        if arr1 is None or arr2 is None or arr1.size == 0 or arr2.size == 0:
            return 0.0
        return np.max(np.abs(arr1 - arr2))

    # Linear inequality
    has_linear_ub = (p_pycutest.aub is not None and p_s2mpj.aub is not None and 
                    p_pycutest.aub.size > 0 and p_s2mpj.aub.size > 0)

    if has_linear_ub:
        aub_match = np.allclose(p_pycutest.aub, p_s2mpj.aub, rtol=rtol, atol=atol)
        bub_match = np.allclose(p_pycutest.bub, p_s2mpj.bub, rtol=rtol, atol=atol)
        
        aub_diff = safe_max_diff(p_pycutest.aub, p_s2mpj.aub)
        bub_diff = safe_max_diff(p_pycutest.bub, p_s2mpj.bub)
        
        print(f"{'✓' if aub_match else '✗'} A_ub: max_diff={aub_diff:.2e}")
        print(f"{'✓' if bub_match else '✗'} b_ub: max_diff={bub_diff:.2e}")
        
        if not (aub_match and bub_match):
            results['passed'] = False
            results['failures'].append("Linear inequality constraints mismatch")
    else:
        print("✓ No linear inequality constraints")

    # Linear equality
    has_linear_eq = (p_pycutest.aeq is not None and p_s2mpj.aeq is not None and 
                    p_pycutest.aeq.size > 0 and p_s2mpj.aeq.size > 0)

    if has_linear_eq:
        aeq_match = np.allclose(p_pycutest.aeq, p_s2mpj.aeq, rtol=rtol, atol=atol)
        beq_match = np.allclose(p_pycutest.beq, p_s2mpj.beq, rtol=rtol, atol=atol)
        
        aeq_diff = safe_max_diff(p_pycutest.aeq, p_s2mpj.aeq)
        beq_diff = safe_max_diff(p_pycutest.beq, p_s2mpj.beq)
        
        print(f"{'✓' if aeq_match else '✗'} A_eq: max_diff={aeq_diff:.2e}")
        print(f"{'✓' if beq_match else '✗'} b_eq: max_diff={beq_diff:.2e}")
        
        if not (aeq_match and beq_match):
            results['passed'] = False
            results['failures'].append("Linear equality constraints mismatch")
    else:
        print("✓ No linear equality constraints")
    
    # Function evaluations
    print("\nFunction Evaluations (100 random samples):")
    print("-" * 40)
    
    x0 = p_pycutest.x0
    np.random.seed(42)  # For reproducibility
    xs = [x0 + np.random.uniform(-1.0, 1.0, size=x0.shape) for _ in range(100)]
    
    f_mismatches = 0
    cub_mismatches = 0
    ceq_mismatches = 0
    
    max_f_diff = 0.0
    max_cub_diff = 0.0
    max_ceq_diff = 0.0
    
    for i, x in enumerate(xs):
        # Objective function
        f_p = p_pycutest.fun(x)
        f_s = p_s2mpj.fun(x)
        f_diff = abs(f_p - f_s)
        max_f_diff = max(max_f_diff, f_diff)
        
        if not np.isclose(f_p, f_s, atol=1e-6):
            if f_mismatches == 0:  # Print first mismatch details
                print(f"    Sample {i}: f mismatch")
                print(f"        pycutest={f_p}, s2mpj={f_s}, diff={f_diff:.2e}")
            f_mismatches += 1
        
        # Nonlinear inequality constraints
        if p_pycutest.m_nonlinear_ub > 0 and p_pycutest.m_nonlinear_ub == p_s2mpj.m_nonlinear_ub:
            cub_p = p_pycutest.cub(x)
            cub_s = p_s2mpj.cub(x)
            cub_diff = np.linalg.norm(cub_p - cub_s)
            max_cub_diff = max(max_cub_diff, cub_diff)
            
            if not np.allclose(cub_p, cub_s, atol=1e-6):
                if cub_mismatches == 0:
                    print(f"    Sample {i}: c_ub mismatch, norm={cub_diff:.2e}")
                cub_mismatches += 1
        
        # Nonlinear equality constraints
        if p_pycutest.m_nonlinear_eq > 0 and p_pycutest.m_nonlinear_eq == p_s2mpj.m_nonlinear_eq:
            ceq_p = p_pycutest.ceq(x)
            ceq_s = p_s2mpj.ceq(x)
            ceq_diff = np.linalg.norm(ceq_p - ceq_s)
            max_ceq_diff = max(max_ceq_diff, ceq_diff)
            
            if not np.allclose(ceq_p, ceq_s, atol=1e-6):
                if ceq_mismatches == 0:
                    print(f"    Sample {i}: c_eq mismatch, norm={ceq_diff:.2e}")
                ceq_mismatches += 1
    
    f_match = f_mismatches == 0
    print(f"{'✓' if f_match else '✗'} Objective: {f_mismatches}/100 mismatches, max_diff={max_f_diff:.2e}")
    if not f_match:
        results['passed'] = False
        results['failures'].append(f"Objective function mismatch ({f_mismatches}/100 samples)")
    
    if p_pycutest.m_nonlinear_ub > 0:
        cub_match = cub_mismatches == 0
        print(f"{'✓' if cub_match else '✗'} c_ub: {cub_mismatches}/100 mismatches, max_diff={max_cub_diff:.2e}")
        if not cub_match:
            results['passed'] = False
            results['failures'].append(f"Nonlinear inequality constraints mismatch ({cub_mismatches}/100 samples)")
    
    if p_pycutest.m_nonlinear_eq > 0:
        ceq_match = ceq_mismatches == 0
        print(f"{'✓' if ceq_match else '✗'} c_eq: {ceq_mismatches}/100 mismatches, max_diff={max_ceq_diff:.2e}")
        if not ceq_match:
            results['passed'] = False
            results['failures'].append(f"Nonlinear equality constraints mismatch ({ceq_mismatches}/100 samples)")
    
    # Summary
    print(f"\n{'='*60}")
    if results['passed']:
        print("✓ ALL CHECKS PASSED")
    else:
        print("✗ SOME CHECKS FAILED:")
        for failure in results['failures']:
            print(f"  - {failure}")
    print(f"{'='*60}\n")
    
    return results


if __name__ == "__main__":
    pb_names = ['HS67', 'HS68', 'HS69', 'HS85', 'HS88', 'HS89', 'HS90', 'HS91', 'HS92']
    
    # 统计结果
    all_results = {}
    passed_count = 0
    failed_count = 0
    
    for pb_name in pb_names:
        p_pycutest = pycutest_load(pb_name)
        p_s2mpj = s2mpj_load(pb_name)
        
        results = compare_problems(p_pycutest, p_s2mpj)
        all_results[pb_name] = results
        
        if results['passed']:
            passed_count += 1
        else:
            failed_count += 1
    
    # 最终总结
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Total problems tested: {len(pb_names)}")
    print(f"Passed: {passed_count}/{len(pb_names)} ({100*passed_count/len(pb_names):.1f}%)")
    print(f"Failed: {failed_count}/{len(pb_names)} ({100*failed_count/len(pb_names):.1f}%)")
    
    if failed_count > 0:
        print(f"\n{'='*60}")
        print("FAILED PROBLEMS:")
        print("-"*60)
        for pb_name, results in all_results.items():
            if not results['passed']:
                print(f"\n{pb_name}:")
                for failure in results['failures']:
                    print(f"  ✗ {failure}")
    
    print(f"\n{'='*60}")
    if passed_count == len(pb_names):
        print("✓ ALL PROBLEMS PASSED")
    else:
        print(f"✗ {failed_count} PROBLEM(S) FAILED")
    print("="*60)