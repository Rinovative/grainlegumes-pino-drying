function generation_root = resolve_generation_root()
% Resolve the numbered generated-simulation area from the sole storage root.

storage_root = getenv('STORAGE_ROOT');
if isempty(storage_root)
    core_dir = fileparts(mfilename('fullpath'));
    simulation_root = fullfile(core_dir, '..', '..', '..');
    repository_root = fullfile(simulation_root, '..', '..');
    storage_root = fullfile(repository_root, '..', 'storage');
end
storage_root = char(java.io.File(storage_root).getCanonicalPath());
generation_root = fullfile(storage_root, '01_generation');
generation_root = char(java.io.File(generation_root).getCanonicalPath());
end
