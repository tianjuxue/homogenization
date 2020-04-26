parameters from deploy_dns.py

generator = Generator(args)
generator.args.relaxation_parameter = 0.1
generator.args.max_newton_iter = 2000
generator.enable_fast_solve = False
generator.args.n_cells = 8
generator.args.metamaterial_mesh_size = 15
generator.args.fluctuation = False
generator.args.F_list_fixed = [[-0., -0.], [0., disp]]
generator.args.gradient = False if pore_flag < 2 else True
generator.anneal_factors = np.linspace(0, 1, 9)

run_and_save(disp=-0.1, pore_flag=0, name='DNS')

画图的时候energy除以了16，force除以了4