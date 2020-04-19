try:
    generator._anealing_solver_fluctuation(False)
    print(generator.energy_density)
    print(generator.probe_all)
except Exception as e:
    print(e)
    print(generator.energy_density)
    print(generator.probe_all)