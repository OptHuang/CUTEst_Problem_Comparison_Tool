function save_to_hdf5(problem_name, filename)
%SAVE_TO_HDF5 Save problem data to an HDF5 file.

    p = s2mpj_load(problem_name);

    % Randomly sample points around the initial guess
    n_samples = 100;
    rng(1);
    X_samples = repmat(p.x0, 1, n_samples) + 0.1 * randn(p.n, n_samples);

    if exist(filename, 'file')
        delete(filename);
    end

    % Save scalar attributes
    h5create(filename, '/name', [1, 1], 'Datatype', 'string');
    h5write(filename, '/name', string(p.name));
    
    h5create(filename, '/n', [1, 1]);
    h5write(filename, '/n', p.n);
    
    h5create(filename, '/mb', [1, 1]);
    h5write(filename, '/mb', p.mb);
    
    h5create(filename, '/m_linear_ub', [1, 1]);
    h5write(filename, '/m_linear_ub', p.m_linear_ub);
    
    h5create(filename, '/m_linear_eq', [1, 1]);
    h5write(filename, '/m_linear_eq', p.m_linear_eq);
    
    h5create(filename, '/m_nonlinear_ub', [1, 1]);
    h5write(filename, '/m_nonlinear_ub', p.m_nonlinear_ub);
    
    h5create(filename, '/m_nonlinear_eq', [1, 1]);
    h5write(filename, '/m_nonlinear_eq', p.m_nonlinear_eq);
    
    h5create(filename, '/mlcon', [1, 1]);
    h5write(filename, '/mlcon', p.mlcon);
    
    h5create(filename, '/mnlcon', [1, 1]);
    h5write(filename, '/mnlcon', p.mnlcon);
    
    h5create(filename, '/mcon', [1, 1]);
    h5write(filename, '/mcon', p.mcon);
    
    h5create(filename, '/ptype', [1, 1], 'Datatype', 'string');
    h5write(filename, '/ptype', string(p.ptype));

    % Save vector/matrix data
    h5create(filename, '/x0', size(p.x0));
    h5write(filename, '/x0', p.x0);
    
    h5create(filename, '/xl', size(p.xl));
    h5write(filename, '/xl', p.xl);
    
    h5create(filename, '/xu', size(p.xu));
    h5write(filename, '/xu', p.xu);

    if ~isempty(p.aub)
        h5create(filename, '/aub', size(p.aub));
        h5write(filename, '/aub', p.aub);
    end
    
    if ~isempty(p.bub)
        h5create(filename, '/bub', size(p.bub));
        h5write(filename, '/bub', p.bub);
    end
    
    if ~isempty(p.aeq)
        h5create(filename, '/aeq', size(p.aeq));
        h5write(filename, '/aeq', p.aeq);
    end
    
    if ~isempty(p.beq)
        h5create(filename, '/beq', size(p.beq));
        h5write(filename, '/beq', p.beq);
    end

    h5create(filename, '/X_samples', size(X_samples));
    h5write(filename, '/X_samples', X_samples);
    
    % Compute and save function values
    fun_values = zeros(n_samples, 1);
    for i = 1:n_samples
        fun_values(i) = p.fun(X_samples(:, i));
    end
    h5create(filename, '/fun_values', size(fun_values));
    h5write(filename, '/fun_values', fun_values);
    
    % Compute and save constraint values
    if p.m_nonlinear_ub > 0
        cub_values = zeros(p.m_nonlinear_ub, n_samples);
        for i = 1:n_samples
            cub_values(:, i) = p.cub(X_samples(:, i));
        end
        h5create(filename, '/cub_values', size(cub_values));
        h5write(filename, '/cub_values', cub_values);
    end
    
    if p.m_nonlinear_eq > 0
        ceq_values = zeros(p.m_nonlinear_eq, n_samples);
        for i = 1:n_samples
            ceq_values(:, i) = p.ceq(X_samples(:, i));
        end
        h5create(filename, '/ceq_values', size(ceq_values));
        h5write(filename, '/ceq_values', ceq_values);
    end
end