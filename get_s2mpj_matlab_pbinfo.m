function get_s2mpj_matlab_pbinfo()
%GET_S2MPJ_MATLAB_PBINFO collect the information about the specified problems
%   in the S2MPJ problem set in MATLAB format and save it to a csv file.

    pb_names = {'HS67', 'HS68', 'HS69', 'HS85', 'HS88', 'HS89', 'HS90', 'HS91', 'HS92'};

    for i = 1:length(pb_names)
        pb_name = pb_names{i};
        filename = sprintf('%s.h5', pb_name);
        save_to_hdf5(pb_name, filename);
        fprintf('Saved problem %s to %s\n', pb_name, filename);
    end
end